"""PREREG v2 shift family S1 (image degradation) and S5 (object removal stress).

S1 follows the ImageNet-C severity parameters (Hendrycks & Dietterich 2019),
implemented with PIL / numpy so they run anywhere and are deterministic:

    gaussian_blur   sigma          [1, 2, 3, 4, 6]
    jpeg            quality        [25, 18, 15, 10, 7]
    pixelate        resize factor  [0.6, 0.5, 0.4, 0.3, 0.25]  (BOX down, BOX up)
    gaussian_noise  std on [0, 1]  [0.08, 0.12, 0.18, 0.26, 0.38]
    contrast        factor c       [0.4, 0.3, 0.2, 0.1, 0.05]  (x -> (x - mean) * c + mean)

Noise uses a generator seeded by (image id, severity), so every run produces the
same corrupted image. These values are frozen by the `prereg-v2` tag.

S5 (exploratory, no hypothesis): one supported object category per image is
"removed" by filling all its COCO instance boxes (10% padding) with the image's
mean colour. The claim for that category becomes absent by construction.
"""
from __future__ import annotations

import io
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter

S1_PARAMS: Dict[str, List[float]] = {
    "gaussian_blur": [1, 2, 3, 4, 6],
    "jpeg": [25, 18, 15, 10, 7],
    "pixelate": [0.6, 0.5, 0.4, 0.3, 0.25],
    "gaussian_noise": [0.08, 0.12, 0.18, 0.26, 0.38],
    "contrast": [0.4, 0.3, 0.2, 0.1, 0.05],
}
S1_CONDITIONS: List[str] = [f"{k}_{s}" for k in S1_PARAMS for s in range(1, 6)]


def parse_condition(cond: str) -> Tuple[str, int]:
    kind, sev = cond.rsplit("_", 1)
    sev = int(sev)
    if kind not in S1_PARAMS or not 1 <= sev <= 5:
        raise ValueError(f"unknown S1 condition {cond}")
    return kind, sev


def corrupt(img: Image.Image, cond: str, image_id: int = 0) -> Image.Image:
    """Apply S1 condition `cond` (e.g. "jpeg_3"); "clean" returns the RGB image."""
    img = img.convert("RGB")
    if cond == "clean":
        return img
    kind, sev = parse_condition(cond)
    p = S1_PARAMS[kind][sev - 1]
    if kind == "gaussian_blur":
        return img.filter(ImageFilter.GaussianBlur(radius=p))
    if kind == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(p))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if kind == "pixelate":
        w, h = img.size
        small = img.resize((max(1, int(w * p)), max(1, int(h * p))), Image.BOX)
        return small.resize((w, h), Image.BOX)
    x = np.asarray(img, dtype=np.float32) / 255.0
    if kind == "gaussian_noise":
        rng = np.random.default_rng(int(image_id) * 10 + sev)
        x = x + rng.normal(0.0, p, size=x.shape)
    elif kind == "contrast":
        m = x.mean(axis=(0, 1), keepdims=True)
        x = (x - m) * p + m
    return Image.fromarray((np.clip(x, 0, 1) * 255).round().astype(np.uint8))


def mask_boxes(img: Image.Image, boxes_xywh: Sequence[Sequence[float]], pad: float = 0.1) -> Image.Image:
    """Fill each COCO box (x, y, w, h), enlarged by `pad`, with the image's mean colour."""
    img = img.convert("RGB")
    a = np.asarray(img).copy()
    mean = a.reshape(-1, 3).mean(axis=0).round().astype(np.uint8)
    H, W = a.shape[:2]
    for x, y, w, h in boxes_xywh:
        dx, dy = w * pad, h * pad
        x0, y0 = int(max(0, np.floor(x - dx))), int(max(0, np.floor(y - dy)))
        x1, y1 = int(min(W, np.ceil(x + w + dx))), int(min(H, np.ceil(y + h + dy)))
        a[y0:y1, x0:x1] = mean
    return Image.fromarray(a)


def pick_removal(image_id: int, supported: Sequence[str]) -> str:
    """Deterministic choice of the category to remove for S5."""
    cats = sorted(supported)
    rng = np.random.default_rng(int(image_id))
    return cats[int(rng.integers(len(cats)))]
