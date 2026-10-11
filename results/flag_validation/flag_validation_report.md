# Deterministic flag validation against PTB-XL scp_codes

Evaluated over 21799 records, no LLM call involved (detect_flags is pure feature-threshold logic).

| Flag | Precision | Recall | F1 | Ground-truth prevalence | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|
| `tachycardia_suspected` | 0.045 | 0.786 | 0.086 | 0.0013 | 22 | 463 | 6 | 21308 |
| `bradycardia_suspected` | 0.000 | nan | nan | 0.0000 | 0 | 3868 | 0 | 17931 |
| `irregular_rhythm_suspected` | 0.024 | 0.609 | 0.047 | 0.0059 | 78 | 3112 | 50 | 18559 |
| `wide_qrs_suspected` | 0.376 | 0.531 | 0.440 | 0.1430 | 1656 | 2751 | 1462 | 15930 |

Precision here means: of the records this flag fired on, what fraction actually carry a matching scp_codes diagnosis (see module docstring for the exact code sets per flag). Low precision with high recall (tachycardia_suspected, irregular_rhythm_suspected) means the fixed threshold catches most true cases but is swamped by false positives - the raw statistical condition (e.g. mean heart rate > 100 bpm over 10s) fires far more often than PTB-XL's cardiologists considered diagnostically significant enough to code. wide_qrs_suspected is the only flag with balanced, moderately usable precision and recall.

bradycardia_suspected is a different case from the other two weak flags, not just another instance of over-triggering: its ground-truth prevalence is exactly 0 (SBRAD never appears with nonzero likelihood anywhere in this database, verified directly - not a code bug). This makes its precision 0.000 by mathematical necessity, not by miscalibration - with zero true positives possible, every one of its 3868 firings is unavoidably a false positive regardless of what threshold is used, and recall is undefined (0/0) rather than merely low. Unlike tachycardia_suspected, there is no threshold sweep that can improve this flag's precision against this dataset - see sweep_threshold's docstring for why bradycardia_suspected is deliberately excluded from it. This flag's real-world usefulness is simply unvalidated here, not demonstrated poor.