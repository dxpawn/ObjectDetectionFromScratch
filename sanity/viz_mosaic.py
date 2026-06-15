"""Gate: render mosaic-augmented samples with their boxes to confirm the mosaic
coordinate math keeps boxes aligned after the 4-image collage + letterbox."""

import os
import sys

import torch
from PIL import ImageDraw
from torchvision.transforms.functional import to_pil_image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import DetectionDataset, load_classes
from utils.transforms import IMAGENET_MEAN, IMAGENET_STD, TrainTransform

COLORS = [(255, 0, 0), (0, 200, 0), (0, 128, 255), (255, 0, 255), (255, 165, 0)]


def denorm(t):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t * std + mean).clamp(0, 1)


def main():
    classes = load_classes("public/classes.json")
    ds = DetectionDataset("public/annotations/train.json", "public/train/images", classes,
                          transform=TrainTransform(640), mosaic_prob=1.0, img_size=640)
    out = os.path.join("sanity", "out")
    os.makedirs(out, exist_ok=True)
    for i in range(4):
        img, boxes, labels, meta = ds[i]
        pil = to_pil_image(denorm(img))
        draw = ImageDraw.Draw(pil)
        for box, lbl in zip(boxes.tolist(), labels.tolist()):
            c = COLORS[lbl % len(COLORS)]
            draw.rectangle(box, outline=c, width=2)
            draw.text((box[0] + 2, box[1] + 2), classes[lbl], fill=c)
        pil.save(os.path.join(out, f"mosaic_{i}.png"))
        print(f"mosaic_{i}: {len(boxes)} boxes, size={tuple(img.shape[1:])}, "
              f"labels={sorted(set(labels.tolist()))}")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
