"""Two-view distillation objectives (CRLD, VRM) and their augmentation, on the GPU.

CRLD: Zhang et al., "Cross-View Consistency Regularisation for Knowledge Distillation", ACM MM 2024
      (github.com/arcaninez/crld, mdistiller/distillers/CRLD.py).
VRM:  Zhang et al., "VRM: Knowledge Distillation via Virtual Relation Matching", ICCV 2025
      (github.com/VISION-SJTU/VRM, mdistiller/distillers/VRM.py).
Both train on a weak view (RandomCrop(32, padding=4) with zero padding + horizontal flip) and a strong view (an
independent crop + flip followed by RandAugment(2, 10) of mdistiller/dataset/cifar100.py: two operations drawn with
replacement from 14, uniform magnitudes, then a Cutout of side U(0, 0.5) x image side filled with (125, 123, 114)).

`randaugment` re-implements that PIL pipeline on uint8-valued float tensors, per sample, following Pillow's
arithmetic (integer luma, blend rounding, the equalize / autocontrast look-up tables, nearest-neighbour affine
sampling at pixel centres). tests/test_randaugment.py compares every operation with the PIL original.
Images are passed as float tensors with values in [0, 255] (B, 3, H, W); any square size works, so the same
augmentation applies to EP-KD's restricted views.
"""
import math

import torch
import torch.nn.functional as F

CUTOUT_FILL = (125.0, 123.0, 114.0)
# (name, min, max) exactly as augment_list() in mdistiller/dataset/cifar100.py (CRLD and VRM repositories)
AUGMENT_LIST = [("AutoContrast", 0, 1), ("Brightness", 0.05, 0.95), ("Color", 0.05, 0.95), ("Contrast", 0.05, 0.95),
                ("Equalize", 0, 1), ("Identity", 0, 1), ("Posterize", 4, 8), ("Rotate", -30, 30),
                ("Sharpness", 0.05, 0.95), ("ShearX", -0.3, 0.3), ("ShearY", -0.3, 0.3), ("Solarize", 0, 256),
                ("TranslateX", -0.3, 0.3), ("TranslateY", -0.3, 0.3)]


# ----------------------------------------------------------------------------- Pillow-equivalent operations
def _blend(deg, img, v):
    """Image.blend(deg, img, v) for uint8 images: deg + v * (img - deg), rounded and clipped (ImagingBlend)."""
    out = deg + v.view(-1, 1, 1, 1) * (img - deg)
    return out.floor().clamp(0, 255)  # (UINT8) truncation of the float result


def _luma(img):
    """Pillow RGB -> L: (R*19595 + G*38470 + B*7471 + 0x8000) >> 16."""
    r, g, b = img[:, 0], img[:, 1], img[:, 2]
    return torch.floor((r * 19595 + g * 38470 + b * 7471 + 32768) / 65536)


def _lut(img, lut):
    """Per-sample, per-channel look-up table: lut (B, 3, 256)."""
    b, c, h, w = img.shape
    idx = img.long().clamp(0, 255).view(b, c, h * w)
    return lut.gather(2, idx).view(b, c, h, w).to(img.dtype)


def _hist(img):
    b, c, h, w = img.shape
    idx = img.long().clamp(0, 255).view(b * c, h * w)
    hist = torch.zeros(b * c, 256, device=img.device, dtype=torch.long)
    hist.scatter_add_(1, idx, torch.ones_like(idx))
    return hist.view(b, c, 256)


def autocontrast(img, v=None):
    """ImageOps.autocontrast(img) (cutoff 0): per channel, map [lo, hi] linearly to [0, 255]."""
    h = _hist(img)
    ar = torch.arange(256, device=img.device)
    present = h > 0
    lo = torch.where(present, ar, 256).amin(2)
    hi = torch.where(present, ar, -1).amax(2)
    den = (hi - lo).clamp(min=1).double()
    scale = torch.full_like(den, 255.0) / den  # true division (scalar / tensor uses a reciprocal)
    offset = -lo.double() * scale
    lut = ar.double().view(1, 1, -1) * scale.unsqueeze(-1)
    lut = lut + offset.unsqueeze(-1)  # separate kernels: no fused multiply-add, same rounding as Python
    lut = torch.where(lut < 0, torch.zeros_like(lut), torch.where(lut > 255, torch.full_like(lut, 255), lut.floor()))
    ident = ar.view(1, 1, -1).expand_as(lut).double()
    lut = torch.where((hi > lo).unsqueeze(-1), lut, ident)
    return _lut(img, lut.long())


def brightness(img, v):
    return _blend(torch.zeros_like(img), img, v)


def color(img, v):
    return _blend(_luma(img).unsqueeze(1).expand_as(img), img, v)


def contrast(img, v):
    mean = torch.floor(_luma(img).mean((1, 2)) + 0.5)  # int(ImageStat(img.convert("L")).mean + 0.5)
    return _blend(mean.view(-1, 1, 1, 1).expand_as(img), img, v)


def equalize(img, v=None):
    """ImageOps.equalize(img): Pillow's per-band look-up table."""
    h = _hist(img)  # (B, 3, 256)
    ar = torch.arange(256, device=img.device)
    nz = h > 0
    last = torch.where(nz, ar, -1).amax(2)  # last non-empty bin
    h_last = h.gather(2, last.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    step = (h.sum(2) - h_last) // 255
    n0 = (step // 2).unsqueeze(-1)
    cum = torch.cumsum(h, 2) - h  # counts strictly before bin i
    lut = (n0 + cum) // step.clamp(min=1).unsqueeze(-1)
    lut = lut.clamp(0, 255)
    ok = (nz.sum(2) > 1) & (step > 0)
    lut = torch.where(ok.unsqueeze(-1), lut, ar.view(1, 1, -1).expand_as(lut))
    return _lut(img, lut)


def identity(img, v=None):
    return img


def posterize(img, v):
    bits = v.long().clamp(min=1)  # int(v), max(1, v)
    mask = (~((1 << (8 - bits)) - 1)) & 255
    return (img.long() & mask.view(-1, 1, 1, 1)).to(img.dtype)


def solarize(img, v):
    return torch.where(img < v.view(-1, 1, 1, 1), img, 255 - img)


def sharpness(img, v):
    """ImageEnhance.Sharpness: blend with ImageFilter.SMOOTH (3x3 kernel [1 1 1; 1 5 1; 1 1 1] / 13, border kept)."""
    k = torch.ones(3, 3, device=img.device, dtype=img.dtype)
    k[1, 1] = 5.0
    k = (k / 13.0).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    sm = F.conv2d(img, k, groups=3)
    sm = (sm + 0.5).floor().clamp(0, 255)  # ImagingFilter rounds to uint8
    deg = img.clone()
    deg[:, :, 1:-1, 1:-1] = sm
    return _blend(deg, img, v)


def _affine(img, m, fill=0.0):
    """Image.transform(AFFINE, (a, b, c, d, e, f), NEAREST) as Pillow computes it (Geometry.c, affine_fixed):
    16.16 fixed point, x_in = (FIX(c + a/2 + b/2) + x FIX(a) + y FIX(b)) >> 16 and likewise for y; output pixels
    whose source falls outside the image get `fill`. m: (B, 6) float64."""
    bsz, ch, h, w = img.shape
    m = m.double()
    fix = lambda v: torch.floor(v * 65536.0 + 0.5).long()
    a0, a1, a3, a4 = fix(m[:, 0]), fix(m[:, 1]), fix(m[:, 3]), fix(m[:, 4])
    a2 = fix(m[:, 2] + m[:, 0] * 0.5 + m[:, 1] * 0.5)
    a5 = fix(m[:, 5] + m[:, 3] * 0.5 + m[:, 4] * 0.5)
    ys = torch.arange(h, device=img.device).view(1, h, 1)
    xs = torch.arange(w, device=img.device).view(1, 1, w)
    v = lambda t: t.view(-1, 1, 1)
    xi = (v(a2) + xs * v(a0) + ys * v(a1)) >> 16
    yi = (v(a5) + xs * v(a3) + ys * v(a4)) >> 16
    valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    flat = (yi.clamp(0, h - 1) * w + xi.clamp(0, w - 1)).view(bsz, 1, h * w).expand(-1, ch, -1)
    out = img.reshape(bsz, ch, h * w).gather(2, flat).view(bsz, ch, h, w)
    return torch.where(valid.unsqueeze(1), out, torch.full_like(out, fill))


def _geo_matrix(name, v, h, w):
    """Pillow's affine matrix (output -> input) of the geometric operations, per sample; v float64 (B,)."""
    z, o = torch.zeros_like(v), torch.ones_like(v)
    if name == "Rotate":  # Image.rotate(v): counter-clockwise about the centre
        a = -torch.deg2rad(v % 360.0)
        cos = torch.round(torch.cos(a) * 1e15) / 1e15  # round(math.cos(angle), 15)
        sin = torch.round(torch.sin(a) * 1e15) / 1e15
        cx, cy = w / 2.0, h / 2.0
        return torch.stack([cos, sin, cos * (-cx) + sin * (-cy) + cx, -sin, cos, -sin * (-cx) + cos * (-cy) + cy], 1)
    if name == "ShearX":
        return torch.stack([o, v, z, z, o, z], 1)
    if name == "ShearY":
        return torch.stack([o, z, z, v, o, z], 1)
    if name == "TranslateX":
        return torch.stack([o, z, v * w, z, o, z], 1)
    if name == "TranslateY":
        return torch.stack([o, z, z, z, o, v * h], 1)
    raise KeyError(name)


def rotate(img, v):
    return _affine(img, _geo_matrix("Rotate", v.double(), *img.shape[-2:]))


def shear_x(img, v):
    return _affine(img, _geo_matrix("ShearX", v.double(), *img.shape[-2:]))


def shear_y(img, v):
    return _affine(img, _geo_matrix("ShearY", v.double(), *img.shape[-2:]))


def translate_x(img, v):
    return _affine(img, _geo_matrix("TranslateX", v.double(), *img.shape[-2:]))


def translate_y(img, v):
    return _affine(img, _geo_matrix("TranslateY", v.double(), *img.shape[-2:]))


OPS = {"AutoContrast": autocontrast, "Brightness": brightness, "Color": color, "Contrast": contrast,
       "Equalize": equalize, "Identity": identity, "Posterize": posterize, "Rotate": rotate, "Sharpness": sharpness,
       "ShearX": shear_x, "ShearY": shear_y, "Solarize": solarize, "TranslateX": translate_x,
       "TranslateY": translate_y}


def cutout(img, v, x0u, y0u):
    """Cutout(img, v) of cifar100.py: side v * W; centre (x0u, y0u) ~ U(0, W); ImageDraw.rectangle(x0, y0, x1, y1)
    with x0 = int(max(0, x0u - side/2)), x1 = min(W, x0 + side), both ends inclusive; v <= 0 leaves the image."""
    bsz, ch, h, w = img.shape
    side = v * w
    x0 = torch.clamp(x0u - side / 2.0, min=0).floor()
    y0 = torch.clamp(y0u - side / 2.0, min=0).floor()
    x1 = torch.clamp(x0 + side, max=w)
    y1 = torch.clamp(y0 + side, max=h)
    ar_x = torch.arange(w, device=img.device, dtype=img.dtype).view(1, 1, w)
    ar_y = torch.arange(h, device=img.device, dtype=img.dtype).view(1, h, 1)
    inside = ((ar_x >= x0.view(-1, 1, 1)) & (ar_x <= x1.view(-1, 1, 1).floor()) &
              (ar_y >= y0.view(-1, 1, 1)) & (ar_y <= y1.view(-1, 1, 1).floor()) & (v > 0).view(-1, 1, 1))
    fill = torch.tensor(CUTOUT_FILL, device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
    return torch.where(inside.unsqueeze(1), fill.expand_as(img), img)


def randaugment(img, n=2, gen=None):
    """RandAugment(2, 10) of mdistiller/dataset/cifar100.py on a batch of uint8-valued float images."""
    bsz, dev = img.size(0), img.device
    out = img
    geo = ("Rotate", "ShearX", "ShearY", "TranslateX", "TranslateY")
    h, w = img.shape[-2:]
    eye = torch.tensor([1.0, 0, 0, 0, 1.0, 0], device=dev, dtype=torch.float64).expand(bsz, 6)
    for _ in range(n):
        choice = torch.randint(0, len(AUGMENT_LIST), (bsz,), device=dev, generator=gen)
        u = torch.rand(bsz, device=dev, generator=gen, dtype=torch.float64)
        new = out
        mat = eye
        for k, (name, lo, hi) in enumerate(AUGMENT_LIST):  # every op on the whole batch, selected per sample
            if name == "Identity":                            # (no host synchronisation)
                continue
            val = lo + float(hi - lo) * u
            if name in geo:  # geometric ops share one affine pass (identity matrix = unchanged pixels)
                mat = torch.where((choice == k).view(-1, 1), _geo_matrix(name, val, h, w), mat)
                continue
            new = torch.where((choice == k).view(-1, 1, 1, 1), OPS[name](out, val.to(out.dtype)), new)
        is_geo = sum((choice == k) for k, (nm, _, _) in enumerate(AUGMENT_LIST) if nm in geo).bool()
        new = torch.where(is_geo.view(-1, 1, 1, 1), _affine(out, mat), new)
        out = new
    v = torch.rand(bsz, device=dev, generator=gen) * 0.5
    x0u = torch.rand(bsz, device=dev, generator=gen) * img.size(-1)
    y0u = torch.rand(bsz, device=dev, generator=gen) * img.size(-2)
    return cutout(out, v.to(out.dtype), x0u.to(out.dtype), y0u.to(out.dtype))


def weak_crop_flip(img, pad=4, gen=None):
    """torchvision RandomCrop(size, padding=pad) (zero padding) + RandomHorizontalFlip on uint8-valued floats."""
    bsz, ch, h, w = img.shape
    xp = F.pad(img, (pad, pad, pad, pad), mode="constant", value=0.0)
    off = torch.randint(0, 2 * pad + 1, (bsz, 2), device=img.device, generator=gen)
    ar = torch.arange(h, device=img.device)
    rows = (off[:, 0].view(-1, 1) + ar).view(bsz, 1, h, 1)
    cols = (off[:, 1].view(-1, 1) + torch.arange(w, device=img.device)).view(bsz, 1, 1, w)
    bi = torch.arange(bsz, device=img.device).view(bsz, 1, 1, 1)
    ci = torch.arange(ch, device=img.device).view(1, ch, 1, 1)
    x = xp[bi, ci, rows, cols]
    flip = torch.rand(bsz, device=img.device, generator=gen) < 0.5
    return torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)


# ----------------------------------------------------------------------------- objectives (verbatim ports)
def _kd(s, t, temperature):
    return F.kl_div(F.log_softmax(s / temperature, 1), F.softmax(t / temperature, 1),
                    reduction="none").sum(1) * temperature ** 2


def crld_loss(s_w, s_s, t_w, t_s, y, ce_w=1.0, wv_w=2.0, cv_w=0.5, temperature=4.0, tau_w=0.9, tau_s=0.2):
    """CRLD.forward_train (defaults = CFG.CRLD of the official repository)."""
    s_w, s_s, t_w, t_s = s_w.float(), s_s.float(), t_w.float(), t_s.float()
    mask_w = F.softmax(t_w, 1).max(1)[0].ge(tau_w).float()
    mask_s = F.softmax(t_s, 1).max(1)[0].ge(tau_s).float()
    loss_ce = ce_w * (F.cross_entropy(s_w, y) + F.cross_entropy(s_s, y))
    loss_wv = wv_w * ((_kd(s_w, t_w, temperature) + _kd(s_s, t_s, temperature)) * mask_w).mean()
    loss_cv = cv_w * ((_kd(s_s, t_w, temperature) + _kd(s_w, t_s, temperature)) * mask_s).mean()
    return loss_ce + loss_wv + loss_cv


def _zsnorm_probs(z, temperature=4.0):
    z = z.view(z.size(0), -1)
    z = (z - z.mean(-1, keepdim=True)) / (1e-7 + z.std(-1, keepdim=True))
    return F.softmax(z / temperature, 1)


def _isv(sw, ss, tw, ts, edge_weight=50.0):
    sw, ss, tw, ts = (_zsnorm_probs(z) for z in (sw, ss, tw, ts))
    with torch.no_grad():
        ntd = F.normalize(tw.unsqueeze(0) - ts.unsqueeze(1), p=2, dim=2)
    nsd = F.normalize(sw.unsqueeze(0) - ss.unsqueeze(1), p=2, dim=2)
    return F.smooth_l1_loss(ntd, nsd) * edge_weight


def _icv(sw, ss, tw, ts, edge_weight=50.0):
    sw, ss, tw, ts = (_zsnorm_probs(z) for z in (sw, ss, tw, ts))
    with torch.no_grad():
        ntd = F.normalize((tw.unsqueeze(2) - ts.unsqueeze(1)).permute(1, 2, 0), p=2, dim=2)
    nsd = F.normalize((sw.unsqueeze(2) - ss.unsqueeze(1)).permute(1, 2, 0), p=2, dim=2)
    return edge_weight * F.smooth_l1_loss(ntd, nsd)


def vrm_loss(s_w, s_s, t_w, t_s, y):
    """VRM.forward_train: CE on both views + 64 * inter-sample + 32 * inter-class virtual-relation losses."""
    s_w, s_s, t_w, t_s = s_w.float(), s_s.float(), t_w.float(), t_s.float()
    loss_ce = F.cross_entropy(s_w, y) + F.cross_entropy(s_s, y)
    return loss_ce + 64.0 * _isv(s_w, s_s, t_w, t_s) + 32.0 * _icv(s_w, s_s, t_w, t_s)


LOSSES = {"crld": crld_loss, "vrm": vrm_loss}
