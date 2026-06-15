# Evaluation Metrics

Per-class detection metrics for the shipped model on the validation set.

- Model: anchor-free FCOS, ResNet-101 backbone, built from scratch.
- Inference: multi-scale TTA (512, 640, 768) with horizontal flip (the default config).
- Validation set: 1500 images. Matching: greedy by descending confidence, IoU >= 0.5 (same as the official grader).

## Per-class metrics at the scoring threshold (confidence >= 0.05)

| Class | AP@0.5 | GT | Predictions | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| person | 0.9016 | 1074 | 5880 | 1033 | 4847 | 41 | 0.176 | 0.962 | 0.297 |
| car | 0.8598 | 283 | 2133 | 272 | 1861 | 11 | 0.128 | 0.961 | 0.225 |
| dog | 0.9228 | 206 | 1014 | 201 | 813 | 5 | 0.198 | 0.976 | 0.330 |
| cat | 0.9303 | 176 | 832 | 171 | 661 | 5 | 0.206 | 0.972 | 0.339 |
| chair | 0.6761 | 282 | 2340 | 247 | 2093 | 35 | 0.106 | 0.876 | 0.188 |
| **mean** | **0.8581** | 2021 | 12199 | 1924 | 10275 | 97 | **0.163** | **0.949** | **0.276** |

Note: these are at confidence >= 0.05, the threshold the model emits and the grader uses
for mAP. mAP only cares about the ranking of confidences, so keeping every low-confidence
box can only help it. That is why precision here is low by design (many low-confidence
boxes are kept to complete the precision-recall curve); it does not reflect the model at a
normal operating point. The table below does.

## Per-class best operating point (confidence threshold that maximizes F1)

| Class | Conf threshold | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| person | 0.414 | 0.908 | 0.820 | 0.862 |
| car | 0.313 | 0.813 | 0.799 | 0.806 |
| dog | 0.525 | 0.951 | 0.840 | 0.892 |
| cat | 0.493 | 0.939 | 0.881 | 0.909 |
| chair | 0.330 | 0.691 | 0.649 | 0.669 |

## Summary

- mAP@0.5: 0.8581 (multi-scale TTA).
- Strongest classes: cat, dog, person (AP around 0.90 or higher).
- Hardest class: chair (heavy occlusion, high appearance variance, easily confused with
  sofas, which are not one of the five labeled classes).
- FN = ground-truth boxes missed at IoU >= 0.5; FP = predicted boxes that matched no
  ground-truth box.
