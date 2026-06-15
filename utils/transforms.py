"""Image + box augmentation and letterbox resizing.

All geometric transforms keep boxes (xyxy, pixel coords) consistent with the
image. ``letterbox`` records the scale and padding so inference can invert it.
"""

from __future__ import annotations

import random

import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PAD_VALUE = 114  # grey padding, matches common detector convention


def letterbox(
    image: Image.Image, boxes: Tensor, size: int, pad_value: int = PAD_VALUE
) -> tuple[Image.Image, Tensor, float, tuple[int, int]]:
    """Resize keeping aspect ratio to fit ``size``x``size``, pad to square.

    Returns ``(padded_image, boxes, scale, (pad_x, pad_y))`` where boxes are in
    the padded image's coordinate frame.
    """
    w, h = image.size
    scale = min(size / w, size / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = image.resize((new_w, new_h), Image.BILINEAR)

    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas = Image.new("RGB", (size, size), (pad_value, pad_value, pad_value))
    canvas.paste(resized, (pad_x, pad_y))

    if boxes.numel():
        boxes = boxes.clone()
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_x
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_y
    return canvas, boxes, scale, (pad_x, pad_y)


def invert_letterbox(
    boxes: Tensor, scale: float, pad: tuple[int, int], orig_w: int, orig_h: int
) -> Tensor:
    """Map boxes from letterboxed coords back to original image pixels."""
    if boxes.numel() == 0:
        return boxes
    boxes = boxes.clone()
    pad_x, pad_y = pad
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, orig_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, orig_h)
    return boxes


def random_horizontal_flip(
    image: Image.Image, boxes: Tensor, p: float = 0.5
) -> tuple[Image.Image, Tensor]:
    if random.random() >= p:
        return image, boxes
    w = image.size[0]
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if boxes.numel():
        boxes = boxes.clone()
        x1 = boxes[:, 0].clone()
        x2 = boxes[:, 2].clone()
        boxes[:, 0] = w - x2
        boxes[:, 2] = w - x1
    return image, boxes


def random_crop(
    image: Image.Image,
    boxes: Tensor,
    labels: Tensor,
    p: float = 0.5,
    min_frac: float = 0.6,
    max_attempts: int = 20,
) -> tuple[Image.Image, Tensor, Tensor]:
    """Random crop that keeps boxes whose center stays inside the crop."""
    if random.random() >= p or boxes.numel() == 0:
        return image, boxes, labels

    w, h = image.size
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2

    for _ in range(max_attempts):
        cw = int(round(w * random.uniform(min_frac, 1.0)))
        ch = int(round(h * random.uniform(min_frac, 1.0)))
        if cw < 1 or ch < 1:
            continue
        x0 = random.randint(0, w - cw)
        y0 = random.randint(0, h - ch)

        keep = (cx >= x0) & (cx < x0 + cw) & (cy >= y0) & (cy < y0 + ch)
        if keep.sum() == 0:
            continue

        new_boxes = boxes[keep].clone()
        new_boxes[:, [0, 2]] = new_boxes[:, [0, 2]].clamp(x0, x0 + cw) - x0
        new_boxes[:, [1, 3]] = new_boxes[:, [1, 3]].clamp(y0, y0 + ch) - y0
        valid = (new_boxes[:, 2] > new_boxes[:, 0] + 1) & (
            new_boxes[:, 3] > new_boxes[:, 1] + 1
        )
        if valid.sum() == 0:
            continue

        cropped = image.crop((x0, y0, x0 + cw, y0 + ch))
        return cropped, new_boxes[valid], labels[keep][valid]

    return image, boxes, labels


def normalize_to_tensor(image: Image.Image) -> Tensor:
    """PIL RGB -> normalized CHW float tensor (ImageNet stats)."""
    tensor = F.to_tensor(image)
    return F.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


class TrainTransform:
    """Flip -> color jitter -> random crop -> letterbox -> normalize.

    Always letterboxes to a fixed ``img_size`` so a batch can be stacked.
    Multi-scale training is applied at the *batch* level in ``train.py`` (resizing
    the stacked batch + scaling boxes), which keeps every image in a batch the
    same size."""

    def __init__(
        self,
        img_size: int,
        use_color_jitter: bool = True,
        use_random_crop: bool = True,
    ) -> None:
        self.img_size = img_size
        self.jitter = (
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
            if use_color_jitter
            else None
        )
        self.use_random_crop = use_random_crop

    def __call__(
        self, image: Image.Image, boxes: Tensor, labels: Tensor, from_mosaic: bool = False
    ) -> tuple[Tensor, Tensor, Tensor, float, tuple[int, int]]:
        image, boxes = random_horizontal_flip(image, boxes)
        if self.jitter is not None:
            image = self.jitter(image)
        # Mosaics already supply scale/translation variety; skip the extra crop.
        if self.use_random_crop and not from_mosaic:
            image, boxes, labels = random_crop(image, boxes, labels)

        image, boxes, scale, pad = letterbox(image, boxes, self.img_size)
        return normalize_to_tensor(image), boxes, labels, scale, pad


class ValTransform:
    """Letterbox -> normalize (no augmentation)."""

    def __init__(self, img_size: int) -> None:
        self.img_size = img_size

    def __call__(
        self, image: Image.Image, boxes: Tensor, labels: Tensor, from_mosaic: bool = False
    ) -> tuple[Tensor, Tensor, Tensor, float, tuple[int, int]]:
        image, boxes, scale, pad = letterbox(image, boxes, self.img_size)
        return normalize_to_tensor(image), boxes, labels, scale, pad
