"""EP-KD for vision on ImageNet: saliency-ordered selection (no score sort) and unit KD weights (gate off)."""
import torch
import torch.nn as nn

from tgsc_kd.config import TGSCConfig
from tgsc_kd.imagenet_parquet import _select_stage_rows
from tgsc_kd.tgsc import _score_candidates_with_teacher


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.fc = nn.Linear(4, 5)

    def forward(self, x):
        return self.fc(self.conv(x).mean((2, 3)))


def _setup():
    torch.manual_seed(0)
    model = _Tiny().eval()
    x = torch.rand(1, 3, 32, 32)
    boxes = torch.tensor([[[0, 0, 16, 16], [16, 16, 32, 32], [0, 16, 16, 32], [16, 0, 32, 16]]])
    logits = model(x)
    return model, x, boxes, logits


def test_sort_false_keeps_input_order_and_scores():
    model, x, boxes, logits = _setup()
    target = logits.argmax(1)
    b_sorted, s_sorted, _ = _score_candidates_with_teacher(model, logits, target, x, boxes, eval_chunk=8)
    b_plain, s_plain, _ = _score_candidates_with_teacher(model, logits, target, x, boxes, eval_chunk=8, sort=False)
    assert torch.equal(b_plain, boxes)
    # the same scores, sorted vs. input order
    assert torch.allclose(torch.sort(s_plain, dim=1, descending=True).values, s_sorted)
    for i in range(boxes.size(1)):
        j = [k for k in range(boxes.size(1)) if torch.equal(b_sorted[0, k], boxes[0, i])][0]
        assert torch.isclose(s_plain[0, i], s_sorted[0, j])


def test_saliency_selection_takes_first_k_without_low():
    boxes = torch.tensor([[0, 0, 16, 16], [16, 16, 32, 32], [0, 16, 16, 32], [16, 0, 32, 16]])
    scores = torch.tensor([0.2, 0.9, 0.5, 0.1])  # saliency order, scores deliberately unsorted
    rows = _select_stage_rows(sample_id=7, resolution=16, boxes_sorted=boxes, scores_sorted=scores,
                              top_k_crops=3, low_k_crops=0)
    assert [r["rank"] for r in rows] == [0, 1, 2]
    assert [(r["y0"], r["x0"]) for r in rows] == [(0, 0), (16, 16), (0, 16)]
    assert all(r["quality_flag"] == 1 for r in rows)


def test_config_defaults_unchanged():
    cfg = TGSCConfig()
    assert cfg.view_selection == "score" and cfg.kd_gate == "on"
