from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from glob import glob
import hashlib
from io import BytesIO
import json
import math
import random
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm.auto import tqdm
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .config import DataConfig, TGSCConfig
from .tgsc import (
    _LayerTaps,
    _boxes_from_grid,
    _build_windows_by_pool,
    _gradcam_pp_from_taps,
    _gather_top_windows,
    _percentile_scale_t,
    _score_candidates_with_teacher,
    _soft_or_t,
    resolve_stage_stride,
)
from .utils import forward_logits, pick_target_layer


ROOT = Path(__file__).resolve().parents[2]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _temp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _temp_path(path)
    if tmp_path.exists():
        tmp_path.unlink()
    tmp_path.write_text(text)
    tmp_path.replace(path)


def _resolve_glob(pattern: str | None) -> tuple[Path, ...]:
    if not pattern:
        return ()
    search = pattern
    if not Path(pattern).is_absolute():
        search = str(ROOT / pattern)
    return tuple(Path(path).resolve() for path in sorted(glob(search)))


@dataclass(frozen=True)
class ParquetSplitInfo:
    name: str
    paths: tuple[Path, ...]
    row_counts: tuple[int, ...]
    offsets: tuple[int, ...]
    total_rows: int

    @classmethod
    def from_glob(cls, name: str, pattern: str | None, limit_samples: int | None = None) -> "ParquetSplitInfo":
        paths = _resolve_glob(pattern)
        if not paths:
            return cls(name=name, paths=(), row_counts=(), offsets=(), total_rows=0)

        row_counts: list[int] = []
        remaining = None if limit_samples is None else int(limit_samples)
        for path in paths:
            if remaining is not None and remaining <= 0:
                break
            count = int(pq.ParquetFile(path).metadata.num_rows)
            if remaining is not None:
                count = min(count, remaining)
                remaining -= count
            row_counts.append(count)

        used_paths = paths[: len(row_counts)]
        offsets: list[int] = []
        seen = 0
        for count in row_counts:
            offsets.append(seen)
            seen += count
        return cls(
            name=name,
            paths=used_paths,
            row_counts=tuple(row_counts),
            offsets=tuple(offsets),
            total_rows=seen,
        )

    def iter_samples(
        self,
        *,
        columns: Sequence[str],
        start: int = 0,
        stop: int | None = None,
        shard_indices: Sequence[int] | None = None,
        batch_size: int = 32,
    ) -> Iterator[dict[str, Any]]:
        end = self.total_rows if stop is None else min(int(stop), self.total_rows)
        if end <= start:
            return

        selected_shards = shard_indices if shard_indices is not None else range(len(self.paths))
        for shard_idx in selected_shards:
            shard_start = self.offsets[shard_idx]
            shard_rows = self.row_counts[shard_idx]
            shard_end = shard_start + shard_rows
            if shard_end <= start:
                continue
            if shard_start >= end:
                break

            local_start = max(0, start - shard_start)
            local_stop = min(shard_rows, end - shard_start)
            if local_stop <= local_start:
                continue

            parquet_file = pq.ParquetFile(self.paths[shard_idx])
            local_row = 0
            for batch in parquet_file.iter_batches(columns=list(columns), batch_size=batch_size):
                if local_row >= local_stop:
                    break

                batch_len = min(len(batch), local_stop - local_row)
                if batch_len <= 0:
                    break

                if local_row + batch_len <= local_start:
                    local_row += batch_len
                    continue

                start_in_batch = max(0, local_start - local_row)
                length = batch_len - start_in_batch

                row_base = local_row + start_in_batch
                image_struct = batch.column(0).slice(start_in_batch, length)
                label_arr = batch.column(1).slice(start_in_batch, length) if len(batch.columns) > 1 else None
                field_names = {field.name for field in image_struct.type}
                bytes_arr = image_struct.field("bytes").to_pylist() if "bytes" in field_names else [None] * length
                path_arr = image_struct.field("path").to_pylist() if "path" in field_names else [None] * length
                labels = label_arr.to_pylist() if label_arr is not None else [None] * length

                for idx in range(length):
                    row_in_shard = row_base + idx
                    yield {
                        "sample_id": shard_start + row_in_shard,
                        "source_shard": shard_idx,
                        "row_in_shard": row_in_shard,
                        "image_bytes": bytes_arr[idx],
                        "image_path": path_arr[idx],
                        "label": labels[idx],
                    }

                local_row += batch_len


def _load_image_bytes(image_bytes: bytes) -> Image.Image:
    with Image.open(BytesIO(image_bytes)) as image:
        return image.convert("RGB")


def _resize_short_side(image: Image.Image, short_side: int) -> Image.Image:
    return TF.resize(image, short_side, interpolation=InterpolationMode.BILINEAR, antialias=True)


def _image_bytes_to_tensor(image_bytes: bytes, short_side: int) -> torch.Tensor:
    image = _load_image_bytes(image_bytes)
    image = _resize_short_side(image, short_side)
    return TF.to_tensor(image)


def _image_bytes_to_byte_tensor(image_bytes: bytes, short_side: int) -> torch.Tensor:
    image = _load_image_bytes(image_bytes)
    image = _resize_short_side(image, short_side)
    return TF.pil_to_tensor(image)


def _image_bytes_to_eval_tensor(
    image_bytes: bytes,
    *,
    resize_short: int,
    crop_size: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    image = _load_image_bytes(image_bytes)
    image = _resize_short_side(image, resize_short)
    image = TF.center_crop(image, [crop_size, crop_size])
    tensor = TF.to_tensor(image)
    return TF.normalize(tensor, mean=list(mean), std=list(std))


def _write_table(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _temp_path(path)
    if tmp_path.exists():
        tmp_path.unlink()
    pq.write_table(table, tmp_path, compression="zstd")
    tmp_path.replace(path)


def _manifest_schema() -> pa.Schema:
    return pa.schema(
        [
            ("sample_id", pa.int64()),
            ("source_shard", pa.int32()),
            ("row_in_shard", pa.int32()),
            ("label", pa.int64()),
            ("image_path", pa.string()),
        ]
    )


def _candidate_schema() -> pa.Schema:
    return pa.schema(
        [
            ("sample_id", pa.int64()),
            ("y0", pa.int32()),
            ("x0", pa.int32()),
            ("y1", pa.int32()),
            ("x1", pa.int32()),
            ("score", pa.float32()),
        ]
    )


def _stage_index_schema() -> pa.Schema:
    return pa.schema(
        [
            ("sample_id", pa.int64()),
            ("resolution", pa.int16()),
            ("y0", pa.int32()),
            ("x0", pa.int32()),
            ("y1", pa.int32()),
            ("x1", pa.int32()),
            ("score", pa.float32()),
            ("quality_flag", pa.int8()),
            ("rank", pa.int16()),
        ]
    )


def _candidate_part_table(rows: list[dict[str, Any]]) -> pa.Table:
    if not rows:
        return pa.Table.from_arrays([pa.array([], type=field.type) for field in _candidate_schema()], schema=_candidate_schema())
    return pa.Table.from_pylist(rows, schema=_candidate_schema())


def _stage_part_table(rows: list[dict[str, Any]]) -> pa.Table:
    if not rows:
        return pa.Table.from_arrays([pa.array([], type=field.type) for field in _stage_index_schema()], schema=_stage_index_schema())
    return pa.Table.from_pylist(rows, schema=_stage_index_schema())


def sort_stage_table(table: pa.Table) -> pa.Table:
    """Stable sort of a stage-index table by sample_id (within-sample row order preserved)."""
    sample_ids = table["sample_id"].to_numpy(zero_copy_only=False)
    order = np.argsort(sample_ids, kind="stable")
    return table.take(pa.array(order))


def write_sorted_stage_index(table: pa.Table, final_path: Path) -> None:
    table = sort_stage_table(table.cast(_stage_index_schema()))
    sids = table["sample_id"].to_numpy(zero_copy_only=False)
    if sids.size > 1 and not bool(np.all(sids[1:] >= sids[:-1])):
        raise AssertionError(f"stage index {final_path} not sorted after sort")
    _write_table(final_path, table)


def _select_stage_rows(
    *,
    sample_id: int,
    resolution: int,
    boxes_sorted: torch.Tensor,
    scores_sorted: torch.Tensor,
    top_k_crops: int,
    low_k_crops: int,
) -> list[dict[str, Any]]:
    num_candidates = scores_sorted.numel()
    if num_candidates == 0:
        return []

    top_sel = list(range(min(top_k_crops, num_candidates)))
    low_start = max(0, num_candidates - low_k_crops)
    low_candidates = list(range(low_start, num_candidates))

    selected_rows: list[dict[str, Any]] = []
    used = set(top_sel)
    for rank, candidate_idx in enumerate(top_sel):
        y0, x0, y1, x1 = boxes_sorted[candidate_idx].tolist()
        selected_rows.append(
            {
                "sample_id": sample_id,
                "resolution": resolution,
                "y0": int(y0),
                "x0": int(x0),
                "y1": int(y1),
                "x1": int(x1),
                "score": float(scores_sorted[candidate_idx].item()),
                "quality_flag": 1,
                "rank": rank,
            }
        )

    low_rank = 0
    for candidate_idx in reversed(low_candidates):
        if candidate_idx in used:
            continue
        y0, x0, y1, x1 = boxes_sorted[candidate_idx].tolist()
        selected_rows.append(
            {
                "sample_id": sample_id,
                "resolution": resolution,
                "y0": int(y0),
                "x0": int(x0),
                "y1": int(y1),
                "x1": int(x1),
                "score": float(scores_sorted[candidate_idx].item()),
                "quality_flag": -1,
                "rank": low_rank,
            }
        )
        low_rank += 1
    return selected_rows


def pad_stage_batch(batch: Sequence[tuple[torch.Tensor, int, float, int, torch.Tensor, int]]) -> tuple[torch.Tensor, ...]:
    images, labels, scores, qflags, boxes, sample_ids = zip(*batch)
    max_h = max(image.shape[-2] for image in images)
    max_w = max(image.shape[-1] for image in images)
    padded = images[0].new_zeros((len(images), 3, max_h, max_w))
    for idx, image in enumerate(images):
        height, width = image.shape[-2:]
        padded[idx, :, :height, :width] = image
    return (
        padded,
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(scores, dtype=torch.float32),
        torch.tensor(qflags, dtype=torch.int16),
        torch.stack(list(boxes)).float(),
        torch.tensor(sample_ids, dtype=torch.long),
    )


class ImageNetValidationDataset(IterableDataset[tuple[torch.Tensor, int]]):
    def __init__(self, split: ParquetSplitInfo, data_cfg: DataConfig):
        self.split = split
        self.data_cfg = data_cfg

    def __len__(self) -> int:
        return self.split.total_rows

    def __iter__(self) -> Iterator[tuple[torch.Tensor, int]]:
        worker = get_worker_info()
        if worker is None:
            shard_indices = list(range(len(self.split.paths)))
        else:
            shard_indices = list(range(worker.id, len(self.split.paths), worker.num_workers))

        for sample in self.split.iter_samples(columns=("image.bytes", "label"), shard_indices=shard_indices, batch_size=16):
            image_bytes = sample["image_bytes"]
            label = sample["label"]
            if image_bytes is None or label is None or int(label) < 0:
                continue
            tensor = _image_bytes_to_eval_tensor(
                image_bytes,
                resize_short=self.data_cfg.eval_resize_short,
                crop_size=self.data_cfg.eval_crop_size,
                mean=self.data_cfg.mean,
                std=self.data_cfg.std,
            )
            yield tensor, int(label)


class ImageNetPrecomputeDataset(IterableDataset[dict[str, Any]]):
    def __init__(
        self,
        *,
        split: ParquetSplitInfo,
        analysis_resize_short: int,
        start: int,
        stop: int,
        read_batch_size: int,
    ) -> None:
        self.split = split
        self.analysis_resize_short = int(analysis_resize_short)
        self.start = int(start)
        self.stop = int(stop)
        self.read_batch_size = max(1, int(read_batch_size))

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        start = self.start
        stop = self.stop
        if worker is not None:
            total = max(0, stop - start)
            span = math.ceil(total / worker.num_workers) if worker.num_workers > 0 else total
            start = start + worker.id * span
            stop = min(stop, start + span)
        if stop <= start:
            return

        for sample in self.split.iter_samples(
            columns=("image.bytes", "label"),
            start=start,
            stop=stop,
            batch_size=self.read_batch_size,
        ):
            image_bytes = sample["image_bytes"]
            label = sample["label"]
            if image_bytes is None or label is None or int(label) < 0:
                continue
            yield {
                "sample_id": int(sample["sample_id"]),
                "label": int(label),
                "image_tensor": _image_bytes_to_tensor(image_bytes, self.analysis_resize_short),
            }


@dataclass(frozen=True)
class StageIndexArrays:
    sample_ids: np.ndarray
    y0: np.ndarray
    x0: np.ndarray
    y1: np.ndarray
    x1: np.ndarray
    scores: np.ndarray
    quality_flags: np.ndarray
    ranks: np.ndarray
    offsets: np.ndarray

    @classmethod
    def from_table(cls, table: pa.Table, num_train_samples: int) -> "StageIndexArrays":
        """Build per-sample offsets.  BUGFIX: the offsets assume rows are grouped by sample_id in
        ascending order, but stage-index parquets written by the multi-worker precompute are
        interleaved.  Always (stable-)sort by sample_id first so row slices belong to their id; the
        stable sort keeps the within-sample order (top rows by rank, then low rows)."""
        sample_ids = table["sample_id"].to_numpy(zero_copy_only=False).astype(np.int64)
        order: Optional[np.ndarray] = None
        if sample_ids.size > 1 and not bool(np.all(sample_ids[1:] >= sample_ids[:-1])):
            order = np.argsort(sample_ids, kind="stable")
            sample_ids = sample_ids[order]

        def col(name: str) -> np.ndarray:
            arr = table[name].to_numpy(zero_copy_only=False)
            return arr[order] if order is not None else arr

        if sample_ids.size and (sample_ids[0] < 0 or sample_ids[-1] >= num_train_samples):
            raise ValueError(
                f"stage index sample_id range [{sample_ids[0]}, {sample_ids[-1]}] outside [0, {num_train_samples})"
            )
        counts = np.bincount(sample_ids, minlength=num_train_samples)
        offsets = np.empty(num_train_samples + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        return cls(
            sample_ids=sample_ids,
            y0=col("y0"),
            x0=col("x0"),
            y1=col("y1"),
            x1=col("x1"),
            scores=col("score"),
            quality_flags=col("quality_flag"),
            ranks=col("rank"),
            offsets=offsets,
        )

    @classmethod
    def from_parquet(cls, path: Path, num_train_samples: int) -> "StageIndexArrays":
        table = pq.read_table(
            path,
            columns=["sample_id", "y0", "x0", "y1", "x1", "score", "quality_flag", "rank"],
        )
        return cls.from_table(table, num_train_samples)

    def rows_for(self, sample_id: int) -> tuple[int, int]:
        """Row slice [start, end) of ``sample_id``; asserts every row in it belongs to that id."""
        start = int(self.offsets[sample_id])
        end = int(self.offsets[sample_id + 1])
        if end > start:
            owners = self.sample_ids[start:end]
            if not bool(np.all(owners == sample_id)):
                raise AssertionError(
                    f"stage index misaligned: rows [{start}, {end}) requested for sample_id={sample_id} "
                    f"belong to {np.unique(owners).tolist()}"
                )
        return start, end

    def __len__(self) -> int:
        return int(self.sample_ids.shape[0])


@dataclass(frozen=True)
class StageScoreCalibration:
    midpoint: float
    scale: float

    @classmethod
    def from_stage_index(cls, stage_index: StageIndexArrays) -> "StageScoreCalibration":
        log_scores = np.log1p(np.clip(stage_index.scores.astype(np.float64), a_min=0.0, a_max=None))
        positive = log_scores[stage_index.quality_flags > 0]
        negative = log_scores[stage_index.quality_flags < 0]

        if positive.size == 0 or negative.size == 0:
            midpoint = float(np.median(log_scores))
            scale = max(float(np.std(log_scores)), 0.25)
            return cls(midpoint=midpoint, scale=scale)

        pos_median = float(np.median(positive))
        neg_median = float(np.median(negative))
        midpoint = 0.5 * (pos_median + neg_median)
        scale = max(0.5 * abs(pos_median - neg_median), 0.1)
        return cls(midpoint=midpoint, scale=scale)

    def normalize_scalar(self, score: float) -> float:
        log_score = math.log1p(max(0.0, float(score)))
        z = (log_score - self.midpoint) / max(self.scale, 1e-6)
        z = max(-20.0, min(20.0, z))
        return float(1.0 / (1.0 + math.exp(-z)))

    def normalize_array(self, scores: np.ndarray) -> np.ndarray:
        log_scores = np.log1p(np.clip(scores.astype(np.float64), a_min=0.0, a_max=None))
        z = np.clip((log_scores - self.midpoint) / max(self.scale, 1e-6), -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-z))


class ImageNetStageDataset(IterableDataset[tuple[torch.Tensor, int, float, int, torch.Tensor, int]]):
    def __init__(
        self,
        *,
        split: ParquetSplitInfo,
        stage_index: StageIndexArrays,
        score_calibration: StageScoreCalibration,
        crop_size: int,
        analysis_resize_short: int,
        shuffle_buffer: int,
        seed: int,
        epoch: int,
    ):
        self.split = split
        self.stage_index = stage_index
        self.score_calibration = score_calibration
        self.crop_size = int(crop_size)
        self.analysis_resize_short = int(analysis_resize_short)
        self.shuffle_buffer = int(max(0, shuffle_buffer))
        self.seed = int(seed)
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.stage_index)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, int, float, int, torch.Tensor, int]]:
        worker = get_worker_info()
        if worker is None:
            worker_id = 0
            worker_count = 1
            shard_indices = list(range(len(self.split.paths)))
        else:
            worker_id = worker.id
            worker_count = worker.num_workers
            shard_indices = list(range(worker.id, len(self.split.paths), worker.num_workers))

        rng = random.Random(self.seed + self.epoch * 997 + worker_id)
        if len(shard_indices) > 1:
            rng.shuffle(shard_indices)
        raw_iter = self._iter_records(shard_indices)
        effective_buffer = max(1, math.ceil(self.shuffle_buffer / max(1, worker_count)))
        if effective_buffer <= 1:
            yield from raw_iter
            return

        buffer: list[tuple[torch.Tensor, int, float, int, torch.Tensor, int]] = []
        for item in raw_iter:
            buffer.append(item)
            if len(buffer) >= effective_buffer:
                idx = rng.randrange(len(buffer))
                yield buffer.pop(idx)
        while buffer:
            idx = rng.randrange(len(buffer))
            yield buffer.pop(idx)

    def _iter_records(self, shard_indices: Sequence[int]) -> Iterator[tuple[torch.Tensor, int, float, int, torch.Tensor, int]]:
        for sample in self.split.iter_samples(columns=("image.bytes", "label"), shard_indices=shard_indices, batch_size=16):
            sample_id = int(sample["sample_id"])
            start, end = self.stage_index.rows_for(sample_id)
            if start == end:
                continue

            image_bytes = sample["image_bytes"]
            label = int(sample["label"])
            if image_bytes is None or label < 0:
                continue

            base_tensor = _image_bytes_to_byte_tensor(image_bytes, self.analysis_resize_short)
            for idx in range(start, end):
                box = torch.tensor(
                    [
                        int(self.stage_index.x0[idx]),
                        int(self.stage_index.y0[idx]),
                        int(self.stage_index.x1[idx]),
                        int(self.stage_index.y1[idx]),
                    ],
                    dtype=torch.float32,
                )
                yield (
                    base_tensor,
                    label,
                    self.score_calibration.normalize_scalar(float(self.stage_index.scores[idx])),
                    int(self.stage_index.quality_flags[idx]),
                    box,
                    sample_id,
                )


CACHE_FORMAT_VERSION = "imagenet_tgsc_v3_sorted_index"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def teacher_arch_signature(teacher: nn.Module) -> str:
    """Hash of parameter/buffer names and shapes: differs between e.g. ResNet34 and ResNet50 even though
    both are instances of a class named ``ResNet``."""
    items = [f"{name}:{tuple(t.shape)}" for name, t in teacher.state_dict().items()]
    return hashlib.sha1("|".join(items).encode()).hexdigest()[:16]


def teacher_weights_sha256(teacher: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in teacher.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class ImageNetTGSCPreprocessor:
    def __init__(
        self,
        teacher: nn.Module,
        data_cfg: DataConfig,
        tgsc_cfg: TGSCConfig,
        *,
        teacher_name: str | None = None,
        teacher_checkpoint: str | Path | None = None,
    ):
        self.teacher = teacher
        self.teacher_name = teacher_name
        self.teacher_checkpoint = None if teacher_checkpoint is None else Path(teacher_checkpoint)
        self.data_cfg = data_cfg
        self.tgsc_cfg = tgsc_cfg
        self.edge_align = bool(getattr(tgsc_cfg, "edge_align_windows", False))
        self.device = next(self.teacher.parameters()).device
        self.teacher.eval()
        if self.device.type == "cuda":
            self.teacher.to(memory_format=torch.channels_last)
        self.target_layer = pick_target_layer(teacher)
        self.stage_sizes = tuple(int(resolution) for resolution, _epochs in data_cfg.stages)

        self.train_split = ParquetSplitInfo.from_glob("train", data_cfg.train_parquet_glob, data_cfg.max_train_samples)
        self.validation_split = ParquetSplitInfo.from_glob(
            "validation",
            data_cfg.validation_parquet_glob,
            data_cfg.max_val_samples,
        )
        self.test_split = ParquetSplitInfo.from_glob("test", data_cfg.test_parquet_glob, None)

        self.cache_root = Path(tgsc_cfg.cache_dir) / "imagenet_tgsc" / self._build_cache_key()
        self.manifest_train_path = self.cache_root / "manifest_train.parquet"
        self.manifest_validation_path = self.cache_root / "manifest_validation.parquet"
        self.state_path = self.cache_root / "precompute_state.json"
        self.stage_dir = self.cache_root / "stage_indexes"
        self._stage_arrays: dict[int, StageIndexArrays] = {}
        self._score_calibrations: dict[int, StageScoreCalibration] = {}
        self._stage_summaries_printed = False

    def _teacher_identity(self) -> dict[str, Any]:
        # BUGFIX: type(teacher).__name__ is "ResNet" for both ResNet34 and ResNet50, so the RN50 run
        # silently reused a ResNet34-built cache.  Key on the architecture name, a parameter-shape
        # signature and the checkpoint (file sha256 if known, otherwise a hash of the loaded weights).
        if self.teacher_checkpoint is not None and self.teacher_checkpoint.exists():
            weights = "file:" + sha256_file(self.teacher_checkpoint)
        else:
            weights = "state:" + teacher_weights_sha256(self.teacher)
        return {
            "teacher_class": type(self.teacher).__name__,
            "teacher_arch": self.teacher_name or type(self.teacher).__name__,
            "teacher_arch_signature": teacher_arch_signature(self.teacher),
            "teacher_checkpoint_sha256": weights,
        }

    def _build_cache_key(self) -> str:
        payload = {
            "format": CACHE_FORMAT_VERSION,
            **self._teacher_identity(),
            "dataset_type": self.data_cfg.dataset_type,
            "train_paths": [str(path) for path in self.train_split.paths],
            "validation_paths": [str(path) for path in self.validation_split.paths],
            "analysis_resize_short": self.data_cfg.analysis_resize_short,
            "stages": list(self.stage_sizes),
            "tgsc": {
                "use_gt_target": self.tgsc_cfg.use_gt_target,
                "w_cam": self.tgsc_cfg.w_cam,
                "w_sal": self.tgsc_cfg.w_sal,
                "temp": self.tgsc_cfg.temp,
                "gamma": self.tgsc_cfg.gamma,
                "pct_lo": self.tgsc_cfg.pct_lo,
                "pct_hi": self.tgsc_cfg.pct_hi,
                "candidate_stride": self.tgsc_cfg.candidate_stride or self.tgsc_cfg.stride,
                "dynamic_stride": self.tgsc_cfg.dynamic_stride,
                "dynamic_stride_power": self.tgsc_cfg.dynamic_stride_power,
                "k_per_img": self.tgsc_cfg.k_per_img,
                "min_frac": self.tgsc_cfg.min_frac,
                "min_mean": self.tgsc_cfg.min_mean,
                "eval_chunk": self.tgsc_cfg.eval_chunk,
                "top_k_crops": self.tgsc_cfg.top_k_crops,
                "low_k_crops": self.tgsc_cfg.low_k_crops,
            },
        }
        # Edge-aligned candidate windows change the candidate set -> must never reuse a plain-grid cache.
        # Only added when on, so every existing (flag-off) key is unchanged.
        if self.edge_align:
            payload["tgsc"]["edge_align_windows"] = True
        # Saliency-ordered selection (EP-KD for vision) selects different rows; only added when not the default.
        if getattr(self.tgsc_cfg, "view_selection", "score") != "score":
            payload["tgsc"]["view_selection"] = self.tgsc_cfg.view_selection
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
        return f"imagenet_{digest}"

    def _default_state(self) -> dict[str, Any]:
        stage_flags = {str(res): False for res in self.stage_sizes}
        part_counts = {str(res): 0 for res in self.stage_sizes}
        return {
            "version": 2,
            "cache_key": self.cache_root.name,
            "edge_align_windows": self.edge_align,
            "phase": "init",
            "phase_detail": None,
            "complete": False,
            "manifests": {
                "train": False,
                "validation": False,
            },
            "candidate_generation_complete": False,
            "finalized_stage_indexes": stage_flags,
            "processed_train_samples": 0,
            "last_checkpointed_sample": 0,
            "last_checkpoint_at": None,
            "active_chunk": None,
            "last_completed_chunk": None,
            "candidate_parts": part_counts,
            "stage_parts": dict(part_counts),
        }

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return self._default_state()
        loaded = json.loads(self.state_path.read_text())
        state = self._default_state()
        for key, value in loaded.items():
            if key in {"candidate_parts", "stage_parts", "manifests", "finalized_stage_indexes"}:
                continue
            state[key] = value

        state["candidate_parts"].update(
            {str(key): int(value) for key, value in loaded.get("candidate_parts", {}).items()}
        )
        state["stage_parts"].update({str(key): int(value) for key, value in loaded.get("stage_parts", {}).items()})
        state["manifests"].update(
            {str(key): bool(value) for key, value in loaded.get("manifests", {}).items()}
        )
        state["finalized_stage_indexes"].update(
            {str(key): bool(value) for key, value in loaded.get("finalized_stage_indexes", {}).items()}
        )
        return state

    def _save_state(self, state: dict[str, Any]) -> None:
        _atomic_write_text(self.state_path, json.dumps(state, indent=2) + "\n")

    def _checkpoint_state(
        self,
        state: dict[str, Any],
        *,
        phase: str | None = None,
        phase_detail: str | None = None,
    ) -> None:
        if phase is not None:
            state["phase"] = phase
        state["phase_detail"] = phase_detail
        state["last_checkpoint_at"] = _now_iso()
        self._save_state(state)

    def _cache_is_complete(self, state: dict[str, Any]) -> bool:
        if not state.get("complete", False):
            return False
        if not state.get("candidate_generation_complete", False):
            return False
        manifests = state.get("manifests", {})
        if not manifests.get("train", False) or not manifests.get("validation", False):
            return False
        finalized = state.get("finalized_stage_indexes", {})
        if not all(finalized.get(str(stage), False) for stage in self.stage_sizes):
            return False
        if not self.manifest_train_path.exists() or not self.manifest_validation_path.exists():
            return False
        return all(self.stage_index_path(stage).exists() for stage in self.stage_sizes)

    def _print_stage_summaries(self) -> None:
        if self._stage_summaries_printed:
            return
        for crop_size in self.stage_sizes:
            stage_index = StageIndexArrays.from_parquet(
                self.stage_index_path(crop_size),
                self.train_split.total_rows,
            )
            calibration = StageScoreCalibration.from_stage_index(stage_index)
            self._log_stage_summary(crop_size, stage_index, calibration)
        self._stage_summaries_printed = True

    def candidate_dir(self, crop_size: int) -> Path:
        return self.cache_root / "candidates" / f"res{crop_size}"

    def stage_tmp_dir(self, crop_size: int) -> Path:
        return self.stage_dir / "tmp" / f"res{crop_size}"

    def stage_index_path(self, crop_size: int) -> Path:
        return self.stage_dir / f"stage_res{crop_size}.parquet"

    def _build_manifest(self, split: ParquetSplitInfo, out_path: Path, state: dict[str, Any]) -> None:
        if out_path.exists():
            if not state["manifests"].get(split.name, False):
                state["manifests"][split.name] = True
                self._checkpoint_state(state, phase="manifests", phase_detail=split.name)
            return

        tmp_path = _temp_path(out_path)
        if tmp_path.exists():
            tmp_path.unlink()

        self._checkpoint_state(state, phase="manifests", phase_detail=split.name)

        writer: Optional[pq.ParquetWriter] = None
        try:
            for shard_idx, path in enumerate(split.paths):
                parquet_file = pq.ParquetFile(path)
                shard_start = split.offsets[shard_idx]
                shard_rows = split.row_counts[shard_idx]
                row_in_shard = 0
                for batch in parquet_file.iter_batches(columns=["image.path", "label"], batch_size=2048):
                    if row_in_shard >= shard_rows:
                        break
                    batch_len = min(len(batch), shard_rows - row_in_shard)
                    image_struct = batch.column(0).slice(0, batch_len)
                    paths = image_struct.field("path").to_pylist()
                    labels = batch.column(1).slice(0, batch_len).to_pylist()
                    rows = {
                        "sample_id": list(range(shard_start + row_in_shard, shard_start + row_in_shard + batch_len)),
                        "source_shard": [shard_idx] * batch_len,
                        "row_in_shard": list(range(row_in_shard, row_in_shard + batch_len)),
                        "label": labels,
                        "image_path": paths,
                    }
                    table = pa.Table.from_pydict(rows, schema=_manifest_schema())
                    if writer is None:
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        writer = pq.ParquetWriter(tmp_path, _manifest_schema(), compression="zstd")
                    writer.write_table(table)
                    row_in_shard += batch_len
        finally:
            if writer is not None:
                writer.close()
        if writer is None:
            empty = pa.Table.from_arrays(
                [pa.array([], type=field.type) for field in _manifest_schema()],
                schema=_manifest_schema(),
            )
            _write_table(out_path, empty)
        else:
            tmp_path.replace(out_path)
        state["manifests"][split.name] = True
        self._checkpoint_state(state, phase="manifests", phase_detail=split.name)

    def _ensure_manifests(self, state: dict[str, Any]) -> None:
        self._build_manifest(self.train_split, self.manifest_train_path, state)
        self._build_manifest(self.validation_split, self.manifest_validation_path, state)

    def _process_image(
        self,
        *,
        sample_id: int,
        image_tensor: torch.Tensor,
        label: int,
        candidate_rows: dict[int, list[dict[str, Any]]],
        stage_rows: dict[int, list[dict[str, Any]]],
    ) -> None:
        analysis = self.compute_union(image_tensor=image_tensor, label=label)
        self.emit_stage_rows(
            analysis,
            sample_id=sample_id,
            candidate_rows=candidate_rows,
            stage_rows=stage_rows,
        )

    def compute_union(self, *, image_tensor: torch.Tensor, label: int) -> dict[str, Any]:
        """Teacher forward/backward + Grad-CAM++/saliency union map for one image (shared by all stages)."""
        xb = image_tensor.unsqueeze(0).to(self.device, non_blocking=True)
        if self.device.type == "cuda":
            xb = xb.contiguous(memory_format=torch.channels_last)
        xb.requires_grad_(True)
        yb = torch.tensor([label], dtype=torch.long, device=self.device)
        taps = _LayerTaps(self.target_layer)
        self.teacher.zero_grad(set_to_none=True)

        logits = forward_logits(self.teacher, xb)
        probs_full = torch.softmax(logits.detach(), dim=-1)
        top1 = probs_full.argmax(dim=1)
        target_idx = yb if self.tgsc_cfg.use_gt_target else top1

        oh = torch.zeros_like(logits)
        oh.scatter_(1, target_idx.view(-1, 1), 1.0)
        logits.backward(gradient=oh, retain_graph=False)

        _, _, height, width = xb.shape
        g = xb.grad.detach().abs().amax(dim=1, keepdim=True)
        sal = (g - g.amin(dim=(2, 3), keepdim=True)) / (
            g.amax(dim=(2, 3), keepdim=True) - g.amin(dim=(2, 3), keepdim=True) + 1e-8
        )

        A, dY = taps.A, taps.dY
        if A is None or dY is None:
            taps.close()
            raise RuntimeError("Target layer hooks did not capture activations or gradients.")

        cam_small = _gradcam_pp_from_taps(A, dY)
        cam = F.interpolate(cam_small, size=(height, width), mode="bilinear", align_corners=False)
        cam = _percentile_scale_t(cam, lo=self.tgsc_cfg.pct_lo, hi=self.tgsc_cfg.pct_hi)
        sal = _percentile_scale_t(sal, lo=self.tgsc_cfg.pct_lo, hi=self.tgsc_cfg.pct_hi)
        union = _soft_or_t(
            cam,
            sal,
            w_cam=self.tgsc_cfg.w_cam,
            w_sal=self.tgsc_cfg.w_sal,
            temp=self.tgsc_cfg.temp,
        )
        if self.tgsc_cfg.gamma != 1.0:
            union = union.clamp(0, 1).pow(1.0 / max(self.tgsc_cfg.gamma, 1e-6))

        taps.close()
        xb.grad = None
        self.teacher.zero_grad(set_to_none=True)
        return {
            "xb": xb,
            "logits": logits.detach(),
            "target_idx": target_idx,
            "union": union.detach(),
            "height": height,
            "width": width,
            "union_cfg": self._union_signature(),
        }

    def _union_signature(self) -> tuple[Any, ...]:
        cfg = self.tgsc_cfg
        return (cfg.use_gt_target, cfg.w_cam, cfg.w_sal, cfg.temp, cfg.gamma, cfg.pct_lo, cfg.pct_hi)

    def emit_stage_rows(
        self,
        analysis: dict[str, Any],
        *,
        sample_id: int,
        candidate_rows: dict[int, list[dict[str, Any]]],
        stage_rows: dict[int, list[dict[str, Any]]],
    ) -> None:
        """Candidate windows, teacher crop scoring and top/low selection for every stage of this cache."""
        if analysis["union_cfg"] != self._union_signature():
            raise ValueError("union map was computed with different TGSC union settings")
        xb = analysis["xb"]
        logits = analysis["logits"]
        target_idx = analysis["target_idx"]
        union = analysis["union"]
        height = analysis["height"]
        width = analysis["width"]
        for crop_size in self.stage_sizes:
            stage_stride = resolve_stage_stride(self.tgsc_cfg, crop_size, self.stage_sizes)
            mean_union_map, frac_map = _build_windows_by_pool(
                union, crop_size, crop_size, stage_stride, self.edge_align
            )
            if mean_union_map.numel() == 0:
                continue
            yh, xw, _ = _gather_top_windows(
                mean_union_map,
                frac_map,
                k_per_img=self.tgsc_cfg.k_per_img,
                min_frac=self.tgsc_cfg.min_frac,
                min_mean=self.tgsc_cfg.min_mean,
            )
            boxes = _boxes_from_grid(yh, xw, crop_size, crop_size, stage_stride, height, width, self.edge_align)
            if boxes.numel() == 0:
                continue
            boxes_sorted, scores_sorted, _ = _score_candidates_with_teacher(
                self.teacher,
                logits.detach(),
                target_idx,
                xb,
                boxes,
                eval_chunk=self.tgsc_cfg.eval_chunk,
                sort=getattr(self.tgsc_cfg, "view_selection", "score") != "saliency",
            )
            boxes_cpu = boxes_sorted[0].detach().cpu()
            scores_cpu = scores_sorted[0].detach().cpu()
            for box_idx in range(scores_cpu.numel()):
                y0, x0, y1, x1 = boxes_cpu[box_idx].tolist()
                candidate_rows[crop_size].append(
                    {
                        "sample_id": sample_id,
                        "y0": int(y0),
                        "x0": int(x0),
                        "y1": int(y1),
                        "x1": int(x1),
                        "score": float(scores_cpu[box_idx].item()),
                    }
                )
            stage_rows[crop_size].extend(
                _select_stage_rows(
                    sample_id=sample_id,
                    resolution=crop_size,
                    boxes_sorted=boxes_cpu,
                    scores_sorted=scores_cpu,
                    top_k_crops=self.tgsc_cfg.top_k_crops,
                    low_k_crops=self.tgsc_cfg.low_k_crops,
                )
            )

    def _make_precompute_loader(self, *, start: int, stop: int) -> DataLoader[dict[str, Any]]:
        dataset = ImageNetPrecomputeDataset(
            split=self.train_split,
            analysis_resize_short=self.data_cfg.analysis_resize_short,
            start=start,
            stop=stop,
            read_batch_size=self.tgsc_cfg.precompute_batch_size,
        )
        num_workers = max(0, int(self.tgsc_cfg.precompute_workers))
        loader_kwargs: dict[str, Any] = {
            "batch_size": None,
            "num_workers": num_workers,
            "pin_memory": self.device.type == "cuda",
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = 4
        return DataLoader(dataset, **loader_kwargs)

    def _write_part_files(
        self,
        *,
        state: dict[str, Any],
        candidate_rows: dict[int, list[dict[str, Any]]],
        stage_rows: dict[int, list[dict[str, Any]]],
    ) -> None:
        for crop_size in self.stage_sizes:
            if candidate_rows[crop_size]:
                part_idx = int(state["candidate_parts"][str(crop_size)])
                part_path = self.candidate_dir(crop_size) / f"part-{part_idx:05d}.parquet"
                _write_table(part_path, _candidate_part_table(candidate_rows[crop_size]))
                state["candidate_parts"][str(crop_size)] = part_idx + 1
            if stage_rows[crop_size]:
                part_idx = int(state["stage_parts"][str(crop_size)])
                part_path = self.stage_tmp_dir(crop_size) / f"part-{part_idx:05d}.parquet"
                _write_table(part_path, _stage_part_table(stage_rows[crop_size]))
                state["stage_parts"][str(crop_size)] = part_idx + 1

    def _finalize_stage_indexes(self, state: dict[str, Any]) -> None:
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        for crop_size in self.stage_sizes:
            crop_key = str(crop_size)
            final_path = self.stage_index_path(crop_size)
            if final_path.exists():
                if not state["finalized_stage_indexes"].get(crop_key, False):
                    state["finalized_stage_indexes"][crop_key] = True
                    self._checkpoint_state(state, phase="finalizing_stage_indexes", phase_detail=f"res{crop_size}")
                continue
            self._checkpoint_state(state, phase="finalizing_stage_indexes", phase_detail=f"res{crop_size}")
            part_paths = sorted(self.stage_tmp_dir(crop_size).glob("part-*.parquet"))
            tmp_path = _temp_path(final_path)
            if tmp_path.exists():
                tmp_path.unlink()
            if part_paths:
                # BUGFIX: part files (and rows inside each part) are not in sample_id order because the
                # multi-worker precompute loader interleaves samples.  Concatenate and stable-sort by
                # sample_id (keeps per-sample top/low row order) before writing the final index.
                table = pa.concat_tables([pq.read_table(path, schema=_stage_index_schema()) for path in part_paths])
                write_sorted_stage_index(table, final_path)
            else:
                _write_table(final_path, _stage_part_table([]))
            state["finalized_stage_indexes"][crop_key] = True
            self._checkpoint_state(state, phase="finalizing_stage_indexes", phase_detail=f"res{crop_size}")

    def ensure_cache(self) -> None:
        state = self._load_state()
        self._ensure_manifests(state)
        if self._cache_is_complete(state):
            self._print_stage_summaries()
            return

        state["complete"] = False
        state.setdefault("candidate_parts", {str(res): 0 for res in self.stage_sizes})
        state.setdefault("stage_parts", {str(res): 0 for res in self.stage_sizes})
        processed = int(state.get("processed_train_samples", 0))
        chunk_size = max(1, int(self.tgsc_cfg.cache_shard_size))

        if processed > 0 and not state.get("candidate_generation_complete", False):
            print(f"[TGSC] Resuming precompute from sample {processed}/{self.train_split.total_rows} in {self.cache_root}")

        progress = tqdm(
            total=self.train_split.total_rows,
            initial=processed,
            desc="TGSC precompute",
            dynamic_ncols=True,
            leave=True,
        )

        try:
            if not state.get("candidate_generation_complete", False):
                self._checkpoint_state(state, phase="candidates")

                while processed < self.train_split.total_rows:
                    chunk_start = processed
                    stop = min(processed + chunk_size, self.train_split.total_rows)
                    state["active_chunk"] = {
                        "start": chunk_start,
                        "stop": stop,
                    }
                    self._checkpoint_state(state, phase="candidates", phase_detail=f"{chunk_start}:{stop}")

                    candidate_rows = {crop_size: [] for crop_size in self.stage_sizes}
                    stage_rows = {crop_size: [] for crop_size in self.stage_sizes}
                    for sample in self._make_precompute_loader(start=processed, stop=stop):
                        self._process_image(
                            sample_id=int(sample["sample_id"]),
                            image_tensor=sample["image_tensor"],
                            label=int(sample["label"]),
                            candidate_rows=candidate_rows,
                            stage_rows=stage_rows,
                        )
                        processed += 1
                        progress.update(1)
                    self._write_part_files(state=state, candidate_rows=candidate_rows, stage_rows=stage_rows)
                    state["processed_train_samples"] = processed
                    state["last_checkpointed_sample"] = processed
                    state["active_chunk"] = None
                    state["last_completed_chunk"] = {
                        "start": chunk_start,
                        "stop": stop,
                    }
                    self._checkpoint_state(state, phase="candidates", phase_detail=f"{chunk_start}:{stop}")

                state["candidate_generation_complete"] = True
                self._checkpoint_state(state, phase="finalizing_stage_indexes")

            self._finalize_stage_indexes(state)
            state["complete"] = True
            self._checkpoint_state(state, phase="complete")
            self._print_stage_summaries()
        finally:
            progress.close()

    def _load_stage_arrays(self, crop_size: int) -> StageIndexArrays:
        if crop_size not in self._stage_arrays:
            self.ensure_cache()
            stage_index = StageIndexArrays.from_parquet(
                self.stage_index_path(crop_size),
                self.train_split.total_rows,
            )
            self._stage_arrays[crop_size] = stage_index
            calibration = StageScoreCalibration.from_stage_index(stage_index)
            self._score_calibrations[crop_size] = calibration
            if not self._stage_summaries_printed:
                self._log_stage_summary(crop_size, stage_index, calibration)
        return self._stage_arrays[crop_size]

    def _log_stage_summary(
        self,
        crop_size: int,
        stage_index: StageIndexArrays,
        calibration: StageScoreCalibration,
    ) -> None:
        quality = stage_index.quality_flags
        scores = stage_index.scores
        norm_scores = calibration.normalize_array(scores)
        good_mask = quality > 0
        bad_mask = quality < 0
        good_count = int(good_mask.sum())
        bad_count = int(bad_mask.sum())
        stage_stride = resolve_stage_stride(self.tgsc_cfg, crop_size, self.stage_sizes)
        avg_good = float(norm_scores[good_mask].mean()) if good_count else 0.0
        avg_bad = float(norm_scores[bad_mask].mean()) if bad_count else 0.0
        print(
            f"[TGSC] Stage {crop_size}: stride={stage_stride} edge_align={self.edge_align} | "
            f"selected={len(stage_index)} (good={good_count}, bad={bad_count}) | "
            f"avg_score good={avg_good:.3f} bad={avg_bad:.3f}"
        )

    def get_loader(
        self,
        crop_size: int,
        *,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        epoch: int = 0,
        seed: int = 0,
    ) -> DataLoader:
        stage_arrays = self._load_stage_arrays(crop_size)
        score_calibration = self._score_calibrations[crop_size]
        dataset = ImageNetStageDataset(
            split=self.train_split,
            stage_index=stage_arrays,
            score_calibration=score_calibration,
            crop_size=crop_size,
            analysis_resize_short=self.data_cfg.analysis_resize_short,
            shuffle_buffer=max(batch_size * 32, 1024),
            seed=seed,
            epoch=epoch,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=pad_stage_batch,
        )


def make_imagenet_validation_loader(data_cfg: DataConfig) -> DataLoader:
    split = ParquetSplitInfo.from_glob("validation", data_cfg.validation_parquet_glob, data_cfg.max_val_samples)
    dataset = ImageNetValidationDataset(split, data_cfg)
    return DataLoader(
        dataset,
        batch_size=data_cfg.val_batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
    )
