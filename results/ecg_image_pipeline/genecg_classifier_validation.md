# GenECG image classifier - validation (real probabilities, held-out fold 10)

ResNet18 (pretrained) on 754 GenECG train images (+133 val for early stopping, best epoch 2), tested on 217 held-out fold-10 images. Multi-label, threshold 0.5. Compare to the 100% VLM zero-shot baseline (macro F1 0.31, MI 0.00).


| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| CD | 0.508 | 0.769 | 0.612 | 78 |
| HYP | 0.580 | 0.691 | 0.631 | 68 |
| MI | 0.632 | 0.608 | 0.619 | 79 |
| NORM | 0.525 | 0.857 | 0.651 | 49 |
| STTC | 0.825 | 0.612 | 0.703 | 85 |
| **macro** | | | **0.643** | |

Exact-match (all 5 right): **0.267** (n=217)

