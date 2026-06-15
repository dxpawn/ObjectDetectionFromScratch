# Object Detection From Scratch

An anchor-free, FCOS-style object detector built from scratch in PyTorch. It detects
five classes: person, car, dog, cat, and chair.

Every part that is specific to a detector is written by hand: the feature pyramid, the
detection head, the target assignment, the losses, the box decoding, the Non-Maximum
Suppression, the mosaic augmentation, and the test-time augmentation. The only borrowed
component is the ImageNet-pretrained ResNet-101 backbone, which the assignment allows.
No off-the-shelf detector is used (no YOLO, Detectron2, MMDetection, or torchvision
detection model), and `torchvision.ops` is not used.

## Results

| Metric | Value |
|---|---|
| Validation mAP@0.5 | 0.858 (default, multi-scale TTA); 0.851 single-scale |
| Performance points | 20 / 20 |
| Per-class AP (person, car, dog, cat, chair) | 0.902, 0.860, 0.923, 0.930, 0.676 |

Scores are produced by the provided grader at `public/tools/evaluate_predictions.py`.
The course grading scale gives full points at mAP@0.5 of 0.75 or higher. The 0.858
validation result is above that bar.
Refer to `METRICS.md` for more details.

## Quickstart

```bash
pip install -r requirements.txt

# train (saves the best checkpoint to ./models/best.pth)
python train.py \
  --train_data ./public/annotations/train.json --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/

# run inference (writes predictions.json)
python predict.py --image_dir /path/to/images --output predictions.json

# score
python public/tools/evaluate_predictions.py \
  --ground_truth public/annotations/val.json --predictions predictions.json --output score.json
```

If `models/best.pth` is not present, `predict.py` downloads it automatically before
running, so inference works without any manual setup.

## Architecture

The model has three parts:

1. Backbone: an ImageNet-pretrained ResNet-101. It produces three feature maps (C3, C4,
   C5) at strides 8, 16, and 32.
2. Feature Pyramid Network: it builds five pyramid levels (P3 to P7) at strides 8, 16,
   32, 64, and 128.
3. Shared detection head: the same weights are used at every pyramid level. It has a
   classification tower (5 class logits, sigmoid, focal loss), a box tower (4 distances
   to the box sides, decoded with exp, GIoU loss), and a centerness output (1 logit, BCE).

This is an anchor-free FCOS design. Every location on a feature map predicts per-class
scores, four distances to the box sides, and a centerness score. A location is a
positive sample for a ground-truth box when it lies inside a center-sampling region and
the box size matches that pyramid level. When several boxes match, the smallest-area box
wins. At inference, the detection confidence is sigmoid(class) times sigmoid(centerness).

## How it maps to the assignment

Every part below is written by hand. The pretrained backbone is the only exception.

| Rubric area | Implementation |
|---|---|
| 1. Data pipeline | `utils/dataset.py`, `utils/transforms.py`. JSON reader, letterbox resize, ImageNet normalize, multi-object handling. Augmentation: horizontal flip, color jitter, random crop, multi-scale, and a hand-written mosaic. Optional chair oversampling. |
| 2. Model | `models/`. `backbone.py` (pretrained ResNet), `fpn.py` (P3 to P7), `head.py` (shared towers), `detector.py`. Predicts box, class, and confidence. |
| 3. Loss | `utils/losses.py`. Sigmoid focal loss for classification, GIoU loss for localization, and BCE for centerness. FCOS target assignment is in `utils/targets.py`. |
| 4. Inference | `utils/postprocess.py`, `predict.py`. Confidence threshold, per-class NMS (hand-written in `utils/boxes.py`), letterbox inversion back to original pixels, and flip plus multi-scale TTA. |

## Development progression

The validation mAP@0.5 improved as the design was refined:

| Stage | Validation mAP@0.5 |
|---|---|
| ResNet-34 at 512 | 0.776 |
| plus 640px and flip TTA | 0.793 |
| ResNet-50, mosaic, chair oversampling at 640 | 0.837 |
| plus multi-scale TTA (512, 640, 768) | 0.844 |
| ResNet-101, same recipe, multi-scale TTA | 0.858 |

Chair is the hardest class. It has heavy occlusion, high appearance variance, and is
easily confused with sofas, which are not one of the five classes. Oversampling, mosaic,
multi-scale TTA, and the ResNet-101 backbone raised its AP from about 0.51 to about 0.68
over the project. Weighted Box Fusion and EMA were also tried, but they did not improve
the score, so they were not used.

## Environment

Python 3.10 or newer. An NVIDIA GPU with CUDA is recommended. Development used an RTX
3060 12 GB with torch 2.9.1+cu126.

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # Windows PowerShell. Linux or macOS: source .venv/bin/activate
pip install -r requirements.txt
```

If needed, install the PyTorch build that matches your CUDA version from
https://pytorch.org/get-started/locally/.

## Training

The minimal command runs with the defaults (ResNet-34 at 512, 30 epochs, SGD with
warmup and cosine schedule, AMP). To reproduce the shipped 0.858 model:

```bash
python train.py \
  --train_data ./public/annotations/train.json --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/ \
  --backbone resnet101 --img_size 640 --batch_size 8 \
  --multiscale --mosaic_prob 0.5 --oversample_chair
```

Training evaluates the validation mAP@0.5 after each epoch and writes the best
checkpoint to `./models/best.pth`. It takes about 7.5 minutes per epoch on an RTX 3060.
Useful flags: `--epochs`, `--lr` (default 0.01), `--backbone` (resnet18, resnet34,
resnet50, resnet101), `--img_size`, `--multiscale`, `--mosaic_prob`, `--close_mosaic`,
`--oversample_chair`, `--chair_boost`, `--ema`, `--workers`, `--no_amp`, `--resume`.

## Inference

```bash
python predict.py --image_dir /path/to/images --output predictions.json
```

The steps are: apply a confidence threshold, run per-class NMS, and rescale boxes back
to original image pixels. Every image appears exactly once. An image with no detections
gets an empty `"boxes"` list.

By default this runs multi-scale TTA at 512, 640, and 768 with horizontal flip
(validation 0.858). Images are processed in chunks, so this stays within memory on
large test sets. For a faster single-scale run, pass `--tta_scales 640` (validation
0.851, about 3 times faster). Use `--no_tta` to also turn off flipping. The flags
`--checkpoint`, `--conf_thresh`, and `--iou_thresh` can also be set.

### Output format

```json
[
  { "image_id": "img_7fd91a4c2e30.jpg",
    "boxes": [ { "class": "person", "confidence": 0.91, "bbox": [48, 72, 210, 356] } ] }
]
```

Each bbox is [xmin, ymin, xmax, ymax] in original image pixels. Each confidence is a
value from 0 to 1.

## Model weights

`train.py` writes the best model, chosen by validation mAP@0.5, to `./models/best.pth`.
The checkpoint also stores the class list, the backbone name, and the input size.
`predict.py` loads it by default.

The checkpoint is about 221 MB. To keep the submission small, it is not bundled with the
code. Instead, `predict.py` downloads it from a hosted URL the first time it runs, if the
file is not already present. To recreate the checkpoint yourself, run the training
command above.

## Project structure

```
.
|-- public/              # dataset and official grader (provided)
|-- models/              # network code (backbone, fpn, head, detector)
|-- utils/               # dataset, transforms, boxes and NMS, targets, losses, postprocess, evaluate
|-- train.py             # training entry point
|-- predict.py           # inference entry point
|-- requirements.txt
|-- README.md
```

## Development notes

The detector was built one step at a time, with a quick correctness check before each
long run: a box-alignment image, a forward-shape check, a target and loss unit test, and
an overfit test on 50 images (loss near 0, mAP near 1.0). One important finding: a
from-scratch FCOS needs the focal-loss prior bias initialization and an SGD schedule
with warmup and cosine decay. A constant learning rate with Adam does not converge.

This is coursework. `ASSIGNMENT DETAILS.md` is the full specification.
