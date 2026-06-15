"""Dataset reader for the assignment's JSON annotation format.

The annotation file looks like::

    {"classes": [...],
     "images": [{"id", "file_name", "width", "height"}, ...],
     "annotations": [{"image_id", "class", "bbox":[xmin,ymin,xmax,ymax]}, ...]}

Images with no annotations are kept (they are valid training/eval samples).
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from typing import Callable

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

CLASSES = ["person", "car", "dog", "cat", "chair"]


def load_classes(path: str | None = None) -> list[str]:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return list(CLASSES)


class DetectionDataset(Dataset):
    def __init__(
        self,
        annotation_json: str,
        image_dir: str,
        classes: list[str],
        transform: Callable | None = None,
        subset_ids: list[str] | None = None,
        mosaic_prob: float = 0.0,
        img_size: int = 640,
    ) -> None:
        with open(annotation_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.image_dir = image_dir
        self.classes = classes
        self.class_to_idx = {name: i for i, name in enumerate(classes)}
        self.transform = transform
        self.mosaic_prob = mosaic_prob  # prob of building a 4-image mosaic
        self.img_size = img_size

        by_image: dict[str, list[dict]] = defaultdict(list)
        for ann in data["annotations"]:
            by_image[ann["image_id"]].append(ann)

        self.samples: list[dict] = []
        for image in data["images"]:
            image_id = image["id"]
            if subset_ids is not None and image_id not in subset_ids:
                continue
            self.samples.append(
                {
                    "id": image_id,
                    "width": image["width"],
                    "height": image["height"],
                    "annotations": by_image.get(image_id, []),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_raw(self, index: int):
        """Load one image + its boxes (xyxy) + labels, no transform."""
        sample = self.samples[index]
        image = Image.open(os.path.join(self.image_dir, sample["id"])).convert("RGB")
        anns = sample["annotations"]
        if anns:
            boxes = torch.tensor([a["bbox"] for a in anns], dtype=torch.float32)
            labels = torch.tensor(
                [self.class_to_idx[a["class"]] for a in anns], dtype=torch.long
            )
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        return image, boxes, labels

    def _build_mosaic(self, index: int):
        """Combine 4 images into one 2S x 2S mosaic (2x2 grid around a random
        center), merging and clipping their boxes. Hand-written augmentation."""
        s = self.img_size
        canvas = 2 * s
        xc = int(random.uniform(0.5 * s, 1.5 * s))
        yc = int(random.uniform(0.5 * s, 1.5 * s))
        indices = [index] + [random.randint(0, len(self.samples) - 1) for _ in range(3)]

        mosaic = Image.new("RGB", (canvas, canvas), (114, 114, 114))
        m_boxes, m_labels = [], []
        for i, idx in enumerate(indices):
            img, boxes, labels = self._load_raw(idx)
            w, h = img.size
            scale = s / max(w, h)
            nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
            img = img.resize((nw, nh), Image.BILINEAR)
            if boxes.numel():
                boxes = boxes * scale

            if i == 0:        # top-left
                x1a, y1a, x2a, y2a = max(xc - nw, 0), max(yc - nh, 0), xc, yc
                x1b, y1b, x2b, y2b = nw - (x2a - x1a), nh - (y2a - y1a), nw, nh
            elif i == 1:      # top-right
                x1a, y1a, x2a, y2a = xc, max(yc - nh, 0), min(canvas, xc + nw), yc
                x1b, y1b, x2b, y2b = 0, nh - (y2a - y1a), min(nw, x2a - x1a), nh
            elif i == 2:      # bottom-left
                x1a, y1a, x2a, y2a = max(xc - nw, 0), yc, xc, min(canvas, yc + nh)
                x1b, y1b, x2b, y2b = nw - (x2a - x1a), 0, nw, min(nh, y2a - y1a)
            else:             # bottom-right
                x1a, y1a, x2a, y2a = xc, yc, min(canvas, xc + nw), min(canvas, yc + nh)
                x1b, y1b, x2b, y2b = 0, 0, min(nw, x2a - x1a), min(nh, y2a - y1a)

            if x2a <= x1a or y2a <= y1a:
                continue
            mosaic.paste(img.crop((x1b, y1b, x2b, y2b)), (x1a, y1a))
            if boxes.numel():
                b = boxes.clone()
                b[:, [0, 2]] += x1a - x1b  # padw
                b[:, [1, 3]] += y1a - y1b  # padh
                m_boxes.append(b)
                m_labels.append(labels)

        if m_boxes:
            boxes = torch.cat(m_boxes)
            labels = torch.cat(m_labels)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, canvas)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, canvas)
            keep = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
            boxes, labels = boxes[keep], labels[keep]
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        return mosaic, boxes, labels

    def __getitem__(self, index: int):
        sample = self.samples[index]
        use_mosaic = self.mosaic_prob > 0 and random.random() < self.mosaic_prob
        if use_mosaic:
            image, boxes, labels = self._build_mosaic(index)
        else:
            image, boxes, labels = self._load_raw(index)

        orig_w, orig_h = image.size
        scale, pad = 1.0, (0, 0)
        if self.transform is not None:
            image, boxes, labels, scale, pad = self.transform(
                image, boxes, labels, from_mosaic=use_mosaic
            )

        meta = {
            "image_id": sample["id"],
            "orig_w": orig_w,
            "orig_h": orig_h,
            "scale": scale,
            "pad": pad,
        }
        return image, boxes, labels, meta


def collate_fn(batch):
    """Stack images (same size within a batch); keep boxes/labels as lists."""
    images, boxes, labels, metas = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(boxes), list(labels), list(metas)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class InferenceDataset(Dataset):
    """Lists image files in a directory (no annotations needed) for predict.py."""

    def __init__(self, image_dir: str, transform: Callable, files: list[str] | None = None) -> None:
        self.image_dir = image_dir
        self.transform = transform
        self.files = files if files is not None else sorted(
            f
            for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        fn = self.files[index]
        image = Image.open(os.path.join(self.image_dir, fn)).convert("RGB")
        orig_w, orig_h = image.size
        empty_b = torch.zeros((0, 4), dtype=torch.float32)
        empty_l = torch.zeros((0,), dtype=torch.long)
        image, _, _, scale, pad = self.transform(image, empty_b, empty_l)
        meta = {
            "image_id": fn,
            "orig_w": orig_w,
            "orig_h": orig_h,
            "scale": scale,
            "pad": pad,
        }
        return image, meta


def inference_collate(batch):
    images, metas = zip(*batch)
    return torch.stack(images, dim=0), list(metas)
