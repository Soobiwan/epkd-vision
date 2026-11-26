"""Strong ResNet32x4 teachers for the 96-px CUB-200 / Oxford-IIIT Pets tables (EP-KD supplement, 11-26).

Student model and evaluation are exactly those of the Table 7 pipeline: CRD ResNet32x4 (view_rules/code_cubpets/models),
96x96 squashed images, official test split (epkd2/data/<ds>_uint8.pt). What changes is only how the teacher is trained:

  --init cifar     start from the CRD CIFAR-100 ResNet32x4 (ckpt_epoch_240.pth, 79.42 %); the 100-way fc is replaced.
  --kd <ckpt>      function matching (Beyer et al., CVPR 2022, "a good teacher is patient and consistent"): the
                   ImageNet-pretrained ResNet50 fine-tuned on the same train split at 224 px (epkd2/runs/tea_*r50_ft)
                   sees the 224-px version of EXACTLY the same augmented crop the student sees at 96 px (same box, same
                   flip, same mixup); loss = KL(teacher || student) at tau, no labels. Without --kd: cross-entropy
                   with label smoothing on the same augmented batches (mixup labels mixed).
  Augmentation: per-image random-resized crop (scale [--scale-min, 1], ratio [3/4, 4/3]) + flip on the 224-px train
  images, student input = antialiased resize of that crop to 96 px (the stored 96-px images are the same squash,
  mean |diff| 1.5/255); mixup Beta(a, a) per batch (a = --mixup, 0 = off).
  SGD nesterov 0.9, cosine with linear warm-up, AMP. Test accuracy on the 96-px test set every --eval-every epochs and
  every epoch in the last 10 %. Saves latest.pt (resume), best_test.pth, final.pth (state dicts), train.jsonl, done.json.
"""
import argparse, json, math, os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.ops import roi_align

CODE = os.environ.get("CUBPETS_CODE", os.path.dirname(os.path.abspath(__file__)))  # dir with models/ (CRD zoo)
sys.path.insert(0, CODE)
import models as crd_models  # noqa: E402

DATA = os.environ.get("CUBPETS_DATA", "/media/e12056/ASi/TGSC/epkd2/data")  # <ds>_uint8.pt, <ds>224_uint8.pt
CIFAR_R32X4 = os.environ.get("CIFAR_R32X4", "/media/cvgl/data3/TGSC/model/cifar_teachers/resnet32x4_vanilla/ckpt_epoch_240.pth")  # CRD
STATS = {  # (mean, std) as in code_cubpets/train.py DATASETS
    "cub": ((0.4856, 0.4994, 0.4324), (0.2235, 0.2188, 0.2580)),
    "pets": ((0.4783, 0.4458, 0.3957), (0.2607, 0.2553, 0.2633)),
    "cub224": ((0.4856, 0.4994, 0.4324), (0.2287, 0.2241, 0.2627)),
    "pets224": ((0.4783, 0.4459, 0.3957), (0.2643, 0.2588, 0.2669)),
}
NC = {"cub": 200, "pets": 37}

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=["cub", "pets"])
ap.add_argument("--run-dir", required=True)
ap.add_argument("--init", default="cifar", choices=["cifar", "none"])
ap.add_argument("--kd", default="", help="ResNet50 teacher state dict (timm resnet50.a1_in1k head re-sized); '' = CE")
ap.add_argument("--tau", type=float, default=1.0)
ap.add_argument("--epochs", type=int, default=300)
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--lr", type=float, default=0.05)
ap.add_argument("--wd", type=float, default=5e-4)
ap.add_argument("--ls", type=float, default=0.1)
ap.add_argument("--mixup", type=float, default=1.0)
ap.add_argument("--scale-min", type=float, default=0.25)
ap.add_argument("--eval-every", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-epochs-this-launch", type=int, default=0, help="smoke tests: stop after this many epochs")
args = ap.parse_args()

os.makedirs(args.run_dir, exist_ok=True)
torch.manual_seed(args.seed)
dev = "cuda"
torch.backends.cudnn.benchmark = True
json.dump(vars(args), open(os.path.join(args.run_dir, "config.json"), "w"), indent=1)

d96 = torch.load(f"{DATA}/{args.dataset}_uint8.pt", map_location="cpu", weights_only=False)
d224 = torch.load(f"{DATA}/{args.dataset}224_uint8.pt", map_location="cpu", weights_only=False)
assert torch.equal(d96["train_y"], d224["train_y"]) and torch.equal(d96["test_y"], d224["test_y"])
Xtr = d224["train_x"].to(dev)  # uint8 [N,3,224,224]
Ytr = d224["train_y"].to(dev)
Xte = d96["test_x"].to(dev)
Yte = d96["test_y"].to(dev)
N, C = Xtr.size(0), NC[args.dataset]
m96, s96 = (torch.tensor(v, device=dev).view(1, 3, 1, 1) for v in STATS[args.dataset])
m224, s224 = (torch.tensor(v, device=dev).view(1, 3, 1, 1) for v in STATS[args.dataset + "224"])

student = crd_models.build_model("resnet32x4", C)
if args.init == "cifar":
    sd = torch.load(CIFAR_R32X4, map_location="cpu", weights_only=False)["model"]
    fc = crd_models.classifier(student)
    fc_keys = {k for k, v in student.state_dict().items() if v.data_ptr() in (fc.weight.data_ptr(), fc.bias.data_ptr())}
    sd = {k: v for k, v in sd.items() if k not in fc_keys}
    missing, unexpected = student.load_state_dict(sd, strict=False)
    assert set(missing) == fc_keys and not unexpected, (missing, unexpected)
student = student.to(dev).to(memory_format=torch.channels_last)

teacher = None
if args.kd:
    import timm
    teacher = timm.create_model("resnet50.a1_in1k", num_classes=C, pretrained=False)
    teacher.load_state_dict(torch.load(args.kd, map_location="cpu", weights_only=False))
    teacher = teacher.to(dev).eval().to(memory_format=torch.channels_last)
    for p in teacher.parameters():
        p.requires_grad_(False)


def logits_of(m, x):
    out = m(x)
    return out[0] if isinstance(out, (tuple, list)) else out


decay, no_decay = [], []
for n_, p in student.named_parameters():
    (no_decay if p.ndim <= 1 else decay).append(p)
opt = torch.optim.SGD([dict(params=decay, weight_decay=args.wd), dict(params=no_decay, weight_decay=0.0)],
                      lr=args.lr, momentum=0.9, nesterov=True)
scaler = torch.amp.GradScaler()
steps_per_epoch = N // args.bs  # drop_last
total = args.epochs * steps_per_epoch


def lr_at(step):
    w = args.warmup * steps_per_epoch
    if step < w:
        return args.lr * (step + 1) / w
    return 0.5 * args.lr * (1 + math.cos(math.pi * (step - w) / max(1, total - w)))


def rrc_boxes(n, side):
    """Inception-style random-resized-crop boxes (x1, y1, x2, y2) in pixels, n per batch, scale [smin, 1]."""
    area = torch.empty(n, device=dev).uniform_(args.scale_min, 1.0) * side * side
    logr = torch.empty(n, device=dev).uniform_(math.log(3 / 4), math.log(4 / 3))
    r = logr.exp()
    w = (area * r).sqrt().clamp(8, side)
    h = (area / r).sqrt().clamp(8, side)
    x1 = torch.rand(n, device=dev) * (side - w)
    y1 = torch.rand(n, device=dev) * (side - h)
    return torch.stack([x1, y1, x1 + w, y1 + h], 1)


@torch.no_grad()
def make_batch(idx):
    x = Xtr[idx].float() / 255.0
    n = x.size(0)
    boxes = rrc_boxes(n, 224)
    rois = torch.cat([torch.arange(n, device=dev, dtype=torch.float).view(-1, 1), boxes], 1)
    x224 = roi_align(x, rois, output_size=(224, 224), spatial_scale=1.0, sampling_ratio=2, aligned=True)
    flip = torch.rand(n, device=dev) < 0.5
    x224 = torch.where(flip.view(-1, 1, 1, 1), x224.flip(3), x224)
    y = F.one_hot(Ytr[idx], C).float()
    if args.mixup > 0:
        lam = float(torch.distributions.Beta(args.mixup, args.mixup).sample())
        perm = torch.randperm(n, device=dev)
        x224 = lam * x224 + (1 - lam) * x224[perm]
        y = lam * y + (1 - lam) * y[perm]
    x96 = F.interpolate(x224, size=(96, 96), mode="bilinear", antialias=True, align_corners=False)
    return ((x96 - m96) / s96).contiguous(memory_format=torch.channels_last), \
           ((x224 - m224) / s224).contiguous(memory_format=torch.channels_last), y


@torch.no_grad()
def evaluate(model, X, Y, mean, std, bs=250):
    model.eval()
    correct = 0
    for i in range(0, X.size(0), bs):
        x = ((X[i:i + bs].float() / 255.0 - mean) / std).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16):
            correct += (logits_of(model, x).argmax(1) == Y[i:i + bs]).sum().item()
    return 100.0 * correct / X.size(0)


start, best, best_ep = 0, 0.0, 0
lp = os.path.join(args.run_dir, "latest.pt")
if os.path.exists(lp):
    ck = torch.load(lp, map_location="cpu", weights_only=False)
    student.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
    start, best, best_ep = ck["epoch"], ck["best"], ck["best_epoch"]
    torch.set_rng_state(ck["rng"]); torch.cuda.set_rng_state(ck["cuda_rng"])
    print(f"resumed at epoch {start}", flush=True)
elif teacher is not None:
    t_acc = evaluate(teacher, d224["test_x"].to(dev), Yte, m224, s224, bs=100)
    print(f"teacher ResNet50 test acc at 224 px: {t_acc:.2f}", flush=True)
    json.dump(dict(teacher_test_224=t_acc), open(os.path.join(args.run_dir, "teacher_acc.json"), "w"))
if start == 0:
    print(f"init ({args.init}) test acc: {evaluate(student, Xte, Yte, m96, s96):.2f} (random head)", flush=True)

launched = 0
for ep in range(start, args.epochs):
    student.train()
    t0, tot, nb = time.time(), 0.0, 0
    perm = torch.randperm(N, device=dev)
    for b in range(steps_per_epoch):
        step = ep * steps_per_epoch + b
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        idx = perm[b * args.bs:(b + 1) * args.bs]
        x96, x224, y = make_batch(idx)
        with torch.autocast("cuda", dtype=torch.float16):
            s = logits_of(student, x96).float()
            if teacher is not None:
                with torch.no_grad():
                    t = teacher(x224).float()
                loss = F.kl_div(F.log_softmax(s / args.tau, 1), F.log_softmax(t / args.tau, 1),
                                reduction="batchmean", log_target=True) * args.tau ** 2
            else:
                yt = y * (1 - args.ls) + args.ls / C
                loss = -(yt * F.log_softmax(s, 1)).sum(1).mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        tot += loss.item(); nb += 1
    e = ep + 1
    rec = dict(epoch=e, lr=lr_at(e * steps_per_epoch - 1), loss=tot / nb, sec=time.time() - t0)
    if e % args.eval_every == 0 or e > 0.9 * args.epochs or e == args.epochs:
        acc = evaluate(student, Xte, Yte, m96, s96)
        rec["test_acc"] = acc
        if acc > best:
            best, best_ep = acc, e
            torch.save(student.state_dict(), os.path.join(args.run_dir, "best_test.pth"))
        rec["best"], rec["best_epoch"] = best, best_ep
    with open(os.path.join(args.run_dir, "train.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
    torch.save(dict(model=student.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), epoch=e, best=best,
                    best_epoch=best_ep, rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()), lp + ".tmp")
    os.replace(lp + ".tmp", lp)
    launched += 1
    if args.max_epochs_this_launch and launched >= args.max_epochs_this_launch:
        sys.exit(0)

torch.save(student.state_dict(), os.path.join(args.run_dir, "final.pth"))
final = evaluate(student, Xte, Yte, m96, s96)
json.dump(dict(best=best, best_epoch=best_ep, final=final, epochs=args.epochs),
          open(os.path.join(args.run_dir, "done.json"), "w"))
print("done", best, best_ep, final, flush=True)
