"""Gate 1: render augmented images with their transformed boxes to verify the
data pipeline keeps boxes aligned through flip / crop / letterbox / normalize."""

import os
import sys

import torch
from PIL import ImageDraw
from torchvision.transforms.functional import to_pil_image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import DetectionDataset, load_classes
from utils.transforms import IMAGENET_MEAN, IMAGENET_STD, TrainTransform

COLORS = [(255, 0, 0), (0, 200, 0), (0, 128, 255), (255, 0, 255), (255, 165, 0)]


def denorm(t: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t * std + mean).clamp(0, 1)


def main() -> None:
    classes = load_classes("public/classes.json")
    ds = DetectionDataset(
        "public/annotations/train.json",
        "public/train/images",
        classes,
        transform=TrainTransform(512),
    )
    out_dir = os.path.join("sanity", "out")
    os.makedirs(out_dir, exist_ok=True)

    # Pick a spread of samples, preferring ones with several boxes.
    multi = [i for i in range(min(400, len(ds))) if len(ds.samples[i]["annotations"]) >= 2]
    picks = (multi[:6] + [0, 1, 2, 3])[:8]

    for i in picks:
        img, boxes, labels, meta = ds[i]
        pil = to_pil_image(denorm(img))
        draw = ImageDraw.Draw(pil)
        for box, lbl in zip(boxes.tolist(), labels.tolist()):
            color = COLORS[lbl % len(COLORS)]
            draw.rectangle(box, outline=color, width=2)
            draw.text((box[0] + 2, box[1] + 2), classes[lbl], fill=color)
        path = os.path.join(out_dir, f"train_{i}.png")
        pil.save(path)
        print(f"{i:4d} {meta['image_id']}  boxes={len(boxes)}  size={tuple(img.shape[1:])}")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
