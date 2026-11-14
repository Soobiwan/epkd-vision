"""Regression tests for the two audited ImageNet TGSC cache bugs.

Bug 1: stage-index parquets are not sorted by sample_id (multi-worker precompute interleaves samples);
       offsets were built assuming sorted rows, so 99.94% of samples read another image's boxes.
Bug 2: the cache key hashed type(teacher).__name__ ("ResNet" for both ResNet34 and ResNet50).
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tgsc_kd.config import DataConfig, TGSCConfig  # noqa: E402
from tgsc_kd.imagenet_parquet import (  # noqa: E402
    ImageNetStageDataset,
    ImageNetTGSCPreprocessor,
    ParquetSplitInfo,
    StageIndexArrays,
    StageScoreCalibration,
    _stage_index_schema,
    _write_table,
)
from tgsc_kd.modeling import build_model  # noqa: E402

N_SAMPLES = 40
ROWS_PER_SAMPLE = 5  # top-3 + low-2


def _rows_for_id(sid: int) -> list[dict]:
    rows = []
    for k in range(ROWS_PER_SAMPLE):
        q = 1 if k < 3 else -1
        rank = k if k < 3 else k - 3
        rows.append(
            {
                "sample_id": sid,
                "resolution": 112,
                # box coordinates encode the owner id so a misaligned lookup is detectable
                "y0": sid,
                "x0": 2 * sid,
                "y1": sid + 50 + k,
                "x1": 2 * sid + 50 + k,
                "score": float(sid) + 0.01 * k,
                "quality_flag": q,
                "rank": rank,
            }
        )
    return rows


def _interleaved_ids(n: int, workers: int = 6) -> list[int]:
    """Mimic the precompute DataLoader: worker w owns a contiguous span, batches arrive round-robin."""
    span = -(-n // workers)
    spans = [list(range(w * span, min(n, (w + 1) * span))) for w in range(workers)]
    out: list[int] = []
    while any(spans):
        for s in spans:
            if s:
                out.append(s.pop(0))
    return out


def _unsorted_table(ids: list[int]) -> pa.Table:
    rows = [r for sid in ids for r in _rows_for_id(sid)]
    return pa.Table.from_pylist(rows, schema=_stage_index_schema())


def test_unsorted_parquet_gives_correct_rows(tmp_path: Path) -> None:
    ids = _interleaved_ids(N_SAMPLES)
    assert ids != sorted(ids)
    path = tmp_path / "stage_res112.parquet"
    pq.write_table(_unsorted_table(ids), path)

    # demonstrate the old bug: naive bincount/cumsum offsets over unsorted rows are misaligned
    raw = pq.read_table(path)["sample_id"].to_numpy()
    naive_offsets = np.concatenate([[0], np.cumsum(np.bincount(raw, minlength=N_SAMPLES))])
    naive_wrong = sum(
        int(not np.all(raw[naive_offsets[s] : naive_offsets[s + 1]] == s)) for s in range(N_SAMPLES)
    )
    assert naive_wrong > 0

    arrays = StageIndexArrays.from_parquet(path, N_SAMPLES)
    for sid in range(N_SAMPLES):
        start, end = arrays.rows_for(sid)
        assert end - start == ROWS_PER_SAMPLE
        assert np.all(arrays.sample_ids[start:end] == sid)
        assert np.all(arrays.y0[start:end] == sid)
        assert np.all(arrays.x0[start:end] == 2 * sid)
        # within-sample order preserved (stable sort): top ranks 0..2 then low ranks 0..1
        assert arrays.quality_flags[start:end].tolist() == [1, 1, 1, -1, -1]
        assert arrays.ranks[start:end].tolist() == [0, 1, 2, 0, 1]
        np.testing.assert_allclose(arrays.scores[start:end], [sid + 0.01 * k for k in range(5)], rtol=1e-6)


def test_rows_for_asserts_on_misaligned_index() -> None:
    sids = np.array([1, 1, 0, 0], dtype=np.int64)  # unsorted, but offsets built as if sorted
    zeros = np.zeros(4, dtype=np.int32)
    bad = StageIndexArrays(
        sample_ids=sids, y0=zeros, x0=zeros, y1=zeros, x1=zeros,
        scores=np.zeros(4, dtype=np.float32), quality_flags=np.ones(4, dtype=np.int8),
        ranks=np.zeros(4, dtype=np.int16), offsets=np.array([0, 2, 4]),
    )
    with pytest.raises(AssertionError):
        bad.rows_for(0)


def _jpeg_bytes(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    img = Image.fromarray(rng.integers(0, 255, size=(300, 400, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _write_fake_imagenet(root: Path, n: int, shards: int = 3) -> str:
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    per = -(-n // shards)
    for s in range(shards):
        ids = list(range(s * per, min(n, (s + 1) * per)))
        images = pa.array([{"bytes": _jpeg_bytes(i), "path": f"img_{i}.JPEG"} for i in ids], type=image_type)
        labels = pa.array([i % 7 for i in ids], type=pa.int64())
        pq.write_table(pa.table({"image": images, "label": labels}), root / f"train-{s:05d}-of-{shards:05d}.parquet")
    return str(root / "train-*.parquet")


def test_stage_dataset_yields_rows_of_requested_id(tmp_path: Path) -> None:
    glob_pattern = _write_fake_imagenet(tmp_path, N_SAMPLES)
    split = ParquetSplitInfo.from_glob("train", glob_pattern)
    assert split.total_rows == N_SAMPLES
    index_path = tmp_path / "stage.parquet"
    pq.write_table(_unsorted_table(_interleaved_ids(N_SAMPLES)), index_path)
    arrays = StageIndexArrays.from_parquet(index_path, N_SAMPLES)
    ds = ImageNetStageDataset(
        split=split, stage_index=arrays, score_calibration=StageScoreCalibration.from_stage_index(arrays),
        crop_size=112, analysis_resize_short=256, shuffle_buffer=0, seed=0, epoch=1,
    )
    seen: dict[int, int] = {}
    for _img, label, _score, _q, box, sid in ds:
        assert label == sid % 7
        # box = (x0, y0, x1, y1) of a row owned by sid
        assert int(box[1]) == sid and int(box[0]) == 2 * sid
        seen[sid] = seen.get(sid, 0) + 1
    assert seen == {sid: ROWS_PER_SAMPLE for sid in range(N_SAMPLES)}


def _data_cfg(glob_pattern: str, stages=((112, 1),)) -> DataConfig:
    return DataConfig(
        dataset_type="imagenet_parquet", train_parquet_glob=glob_pattern,
        validation_parquet_glob=glob_pattern, test_parquet_glob=None, stages=stages,
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
    )


def test_finalize_sorts_interleaved_parts(tmp_path: Path) -> None:
    glob_pattern = _write_fake_imagenet(tmp_path, N_SAMPLES)
    teacher = build_model("ResNet18", num_classes=1000)
    pre = ImageNetTGSCPreprocessor(
        teacher, _data_cfg(glob_pattern), TGSCConfig(cache_dir=str(tmp_path / "cache")), teacher_name="ResNet18"
    )
    ids = _interleaved_ids(N_SAMPLES)
    # two part files, each internally interleaved, second one containing lower ids too
    for part, chunk in enumerate([ids[: N_SAMPLES // 2], ids[N_SAMPLES // 2 :]]):
        _write_table(pre.stage_tmp_dir(112) / f"part-{part:05d}.parquet", _unsorted_table(chunk))
    state = pre._default_state()
    pre._finalize_stage_indexes(state)
    sids = pq.read_table(pre.stage_index_path(112))["sample_id"].to_numpy()
    assert np.all(np.diff(sids) >= 0)
    assert len(sids) == N_SAMPLES * ROWS_PER_SAMPLE
    arrays = StageIndexArrays.from_parquet(pre.stage_index_path(112), N_SAMPLES)
    for sid in range(N_SAMPLES):
        s, e = arrays.rows_for(sid)
        assert np.all(arrays.y0[s:e] == sid)
        assert arrays.quality_flags[s:e].tolist() == [1, 1, 1, -1, -1]


TEACHER_DIR = ROOT / "model" / "imagenet_teachers"
RN34_CKPT = TEACHER_DIR / "ResNet34_vanilla" / "resnet34-333f7ec4.pth"
RN50_CKPT = TEACHER_DIR / "ResNet50_vanilla" / "resnet50-19c8e357.pth"


def _key(tmp_path: Path, arch: str, ckpt: Path | None, **tgsc_overrides) -> tuple[str, str]:
    glob_pattern = str(tmp_path / "none-*.parquet")
    teacher = build_model(arch, num_classes=1000)
    tgsc = TGSCConfig(cache_dir=str(tmp_path / "cache"), top_k_crops=3, low_k_crops=2, **tgsc_overrides)
    pre = ImageNetTGSCPreprocessor(teacher, _data_cfg(glob_pattern), tgsc, teacher_name=arch, teacher_checkpoint=ckpt)
    return pre.cache_root.name, type(teacher).__name__


def test_cache_key_differs_between_resnet34_and_resnet50(tmp_path: Path) -> None:
    rn34 = RN34_CKPT if RN34_CKPT.exists() else None
    rn50 = RN50_CKPT if RN50_CKPT.exists() else None
    key34, cls34 = _key(tmp_path, "ResNet34", rn34)
    key50, cls50 = _key(tmp_path, "RN50", rn50)
    assert cls34 == cls50 == "ResNet"  # the class name alone cannot tell them apart (old bug)
    assert key34 != key50
    # without checkpoint paths the arch signature + weight hash must still separate them
    assert _key(tmp_path, "ResNet34", None)[0] != _key(tmp_path, "ResNet50", None)[0]
    # deterministic for identical inputs
    assert _key(tmp_path, "ResNet34", rn34)[0] == key34
    # selection settings are part of the key
    assert _key(tmp_path, "ResNet34", rn34, k_per_img=8)[0] != key34


def test_cache_key_differs_for_different_checkpoints_same_arch(tmp_path: Path) -> None:
    a = tmp_path / "a.pth"
    b = tmp_path / "b.pth"
    m = build_model("ResNet34", num_classes=1000)
    torch.save(m.state_dict(), a)
    with torch.no_grad():
        next(m.parameters()).add_(1.0)
    torch.save(m.state_dict(), b)
    assert _key(tmp_path, "ResNet34", a)[0] != _key(tmp_path, "ResNet34", b)[0]
