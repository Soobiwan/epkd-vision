"""Minimal repro: do parameter / buffer updates persist on torch_tpu?  Prints PASS/FAIL per case vs CPU.
python3 bench/tpu_inplace_probe.py"""
import copy
import torch
import torch.nn as nn
import torch_tpu  # noqa: F401

dev = torch.device("tpu")
torch.manual_seed(0)


def report(name, ok, extra=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {extra}", flush=True)


# 1. raw in-place ops on a parameter / buffer living on the device
p = nn.Parameter(torch.ones(8, device=dev))
with torch.no_grad():
    p.add_(1.0)
report("param.add_ under no_grad", torch.allclose(p.cpu(), torch.full((8,), 2.0)))
with torch.no_grad():
    p.view(-1)[:4].mul_(3.0)
report("in-place through a view", torch.allclose(p.cpu()[:4], torch.full((4,), 6.0)), str(p.cpu().tolist()))
b = torch.zeros(8, device=dev)
b.mul_(0.9).add_(torch.ones(8, device=dev), alpha=0.1)
report("buffer mul_/add_ chain", torch.allclose(b.cpu(), torch.full((8,), 0.1)))
ts = [torch.ones(4, device=dev) for _ in range(3)]
torch._foreach_add_(ts, 1.0)
report("_foreach_add_ scalar", all(torch.allclose(t.cpu(), torch.full((4,), 2.0)) for t in ts))
torch._foreach_add_(ts, [torch.ones(4, device=dev)] * 3, alpha=0.5)
report("_foreach_add_ list alpha", all(torch.allclose(t.cpu(), torch.full((4,), 2.5)) for t in ts))

# 2. module moved with .to(): parameter identity, grads, SGD variants vs CPU
base = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 5))
x = torch.randn(16, 3, 12, 12)
y = torch.randint(0, 5, (16,))
ids_before = [id(q) for q in base.parameters()]
moved = copy.deepcopy(base)
ids_cpu = [id(q) for q in moved.parameters()]
moved.to(dev)
report("module.to keeps Parameter objects", [id(q) for q in moved.parameters()] == ids_cpu)


def one_step(model, device, **optkw):
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, **optkw)
    model.train()
    loss = nn.functional.cross_entropy(model(x.to(device)), y.to(device))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gnorm = sum(float(q.grad.abs().sum()) for q in model.parameters() if q.grad is not None)
    ngrad = sum(q.grad is not None for q in model.parameters())
    opt.step()
    return float(loss), gnorm, ngrad


ref = copy.deepcopy(base)
lref, gref, _ = one_step(ref, torch.device("cpu"))
for kw in ({}, {"foreach": False}, {"foreach": True}, {"fused": True}):
    m = copy.deepcopy(base).to(dev)
    try:
        l, g, ng = one_step(m, dev, **kw)
        pd = max((a - b.detach().cpu()).abs().max().item() for a, b in zip(ref.parameters(), m.parameters()))
        upd = max((a - b).abs().max().item() for a, b in zip(ref.parameters(), base.parameters()))
        bd = max((a - b.detach().cpu()).abs().max().item() for a, b in zip(ref.buffers(), m.buffers()) if a.dtype.is_floating_point)
        report(f"SGD step {kw or 'default'}", pd < 1e-3 * max(upd, 1e-6) + 1e-5 and bd < 1e-4,
               f"loss {l:.5f}/{lref:.5f} grad-sum {g:.4f}/{gref:.4f} grads {ng} | max param diff {pd:.2e} (update {upd:.2e}) "
               f"| max buffer diff {bd:.2e}")
    except Exception as e:  # noqa: BLE001
        report(f"SGD step {kw or 'default'}", False, f"ERROR {e!r}"[:300])

# 3. BatchNorm running stats in train mode
bn = nn.BatchNorm2d(8)
bn_t = copy.deepcopy(bn).to(dev)
z = torch.randn(4, 8, 5, 5)
bn.train()(z)
bn_t.train()(z.to(dev))
report("BatchNorm running stats update", torch.allclose(bn.running_mean, bn_t.running_mean.cpu(), atol=1e-5)
       and torch.allclose(bn.running_var, bn_t.running_var.cpu(), atol=1e-5),
       f"mean diff {(bn.running_mean - bn_t.running_mean.cpu()).abs().max():.2e} nbt {int(bn_t.num_batches_tracked.cpu())}")
print("INPLACE_PROBE_DONE", flush=True)
