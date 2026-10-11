# VLM zero-shot validation of the ECG-image branch (100% VLM, no classifier)

Model **qwen2.5vl:7b**, 50 class-stratified GenECG images (variant A, clean), PTB-XL test fold 10, seed 0. Zero-shot 5-superclass presence vs true PTB-XL labels.


| class | precision | recall | F1 | support | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| CD | 0.469 | 0.682 | 0.556 | 22 | 15 | 17 | 7 |
| HYP | 0.600 | 0.200 | 0.300 | 15 | 3 | 2 | 12 |
| MI | 0.000 | 0.000 | 0.000 | 18 | 0 | 0 | 18 |
| NORM | 0.167 | 0.273 | 0.207 | 11 | 3 | 15 | 8 |
| STTC | 0.406 | 0.684 | 0.510 | 19 | 13 | 19 | 6 |
| **macro** | **0.328** | **0.368** | **0.314** | | | | |

Exact-match (all 5 classes right): **0.100**  (n=50)

