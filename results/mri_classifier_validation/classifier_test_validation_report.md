# Learned MRI pathology classifier - end-to-end test validation

Evaluated on 50 held-out test patients (never seen during YOLO/U-Net/classifier training or model selection). Predictions aggregated per patient (mean softmax probability across all of that patient's ED+ES slices, then argmax) - not a slice-level number.

**Overall per-patient accuracy: 0.6200** (31/50), chance level = 0.20 for 5 classes.

## Per-class precision/recall/F1

| Group | Precision | Recall | F1    |N patients|
|-------|-----------|--------|-------|----------|
| DCM   | 0.636     | 0.700  | 0.667 | 10       |
| HCM   | 0.714     | 0.500  | 0.588 | 10       |
| MINF  | 0.400     | 0.600  | 0.480 | 10       |
| NOR   | 0.714     | 0.500  | 0.588 | 10       |
| RV    | 0.800     | 0.800  | 0.800 | 10       |

## Confusion matrix (rows = true, columns = predicted)

| true_group   |   DCM |   HCM |   MINF |   NOR |   RV |
|:-------------|------:|------:|-------:|------:|-----:|
| DCM          |     7 |     0 |      3 |     0 |    0 |
| HCM          |     1 |     5 |      4 |     0 |    0 |
| MINF         |     3 |     1 |      6 |     0 |    0 |
| NOR          |     0 |     1 |      2 |     5 |    2 |
| RV           |     0 |     0 |      0 |     2 |    8 |