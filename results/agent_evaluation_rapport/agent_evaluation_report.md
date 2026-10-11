# Per-agent report evaluation — ECG signal vs ECG image

Same 20 held-out PTB-XL fold-10 records (held out for BOTH the signal embedding and the image classifier). GenECG images = the same PTB-XL records rendered as images, so both branches share the same cardiologist reference report.

**Interpretation caveat:** BLEU/ROUGE are low BY DESIGN — the reference is a diagnostic interpretation, the generated text is a quantitative/qualitative description (diagnosis is forbidden in the agent prompts). The **clinical cosine** (biomedical embedding, NeuML/pubmedbert-base-embeddings) is the meaningful semantic metric. Only ECG has a reference report, so MRI/patient accuracy is their flag validation, not this text table. The ECG SIGNAL branch is run in its single-lead form here for tractability (the multilead+embedding form is much slower); the generated narrative is what's scored.

## ECG SIGNAL branch (n=20, PTB-XL fold 10)

| ecg_id | reference (PTB-XL/RTLP) | generated | BLEU | ROUGE-1 | clin.cosine |
|---|---|---|---:|---:|---:|
| 9823 | sinusrytm vÄnster el-axel vÄnstersidigt skÄnkelblock ÖvervÄg | The ECG analysis shows a regular rhythm with a mean heart ra | 0.00 | 0.000 | 0.301 |
| 2146 | sinus rhythm first degree av block, pr interval 0.24 seconds | The ECG analysis indicates a rhythm with a mean heart rate o | 0.93 | 0.083 | 0.687 |
| 15899 | sinus rhythm location type normal incomplete right bundle br | The ECG analysis indicates a heart rate of 52.1 bpm with a s | 0.00 | 0.000 | 0.496 |
| 17909 | sinus rhythm position type normal r-s # transition in v deri | The ECG shows a heart rate of 58.1 bpm, with a mean RR inter | 0.95 | 0.063 | 0.584 |
| 10180 | sinus rhythm. voltages are high in chest leads suggesting lv | The ECG analysis shows a duration of 10 seconds, with 10 QRS | 0.80 | 0.000 | 0.635 |
| 3092 | premature ventricular contraction(s). sinus rhythm. voltages | The ECG analysis shows a heart rate of 56.2 bpm with a stand | 0.87 | 0.078 | 0.690 |
| 1745 | sinus rhythm position type normal atrial overload left possi | The ECG analysis shows a duration of 10 seconds, 12 QRS comp | 0.46 | 0.023 | 0.385 |
| 8083 | sinusrhythm location type normal moderate amplitude criteria | The ECG analysis shows a heart rate of 58.1 bpm, with a mean | 0.00 | 0.000 | 0.405 |
| 15400 | sinus rhythm suspected p-sinistrocardiale location type norm | The ECG analysis indicates a duration of 10 seconds, 10 QRS  | 1.25 | 0.100 | 0.646 |
| 11664 | sinus arrhythmia p-sinistrocardiale location type normal qrs | The ECG analysis reveals a regular rhythm with a mean heart  | 0.00 | 0.024 | 0.572 |
| 16905 | atrial fibrillation. st segments are depressed in v4,5,6. t  | The ECG analysis shows a regular rhythm with a mean heart ra | 1.19 | 0.092 | 0.558 |
| 3122 | sinus rhythm. q waves in ii, iii, avf consistent with old in | The ECG analysis shows a regular sinus rhythm with a mean he | 1.48 | 0.101 | 0.632 |
| 5975 | sinusrhythm position type normal normal ecg | The ECG analysis reveals a regular rhythm with a mean heart  | 0.00 | 0.032 | 0.610 |
| 15548 | sinus bradycardia position type normal normal ekg | The ECG shows a heart rate of 57.5 bpm with a standard devia | 0.57 | 0.050 | 0.570 |
| 2596 | sinus rhythm position type normal left posterior hemiblock | The ECG analysis indicates a regular rhythm with a mean hear | 0.52 | 0.027 | 0.439 |
| 19256 | sinus bradycardia position type normal normal ekg | The ECG shows an irregular rhythm with a heart rate of 55.5  | 0.00 | 0.000 | 0.610 |
| 3508 | sinus rhythm and junctional rhythm. non-specific t wave flat | The ECG analysis shows an irregular rhythm with a heart rate | 0.92 | 0.083 | 0.692 |
| 8533 | sinus rhythm location type normal path. repolarization proba | The ECG analysis reveals a duration of 10 seconds, with 15 Q | 0.56 | 0.000 | 0.437 |
| 8450 | sinus rhythm. low limb lead voltage. st segments are depress | The ECG analysis indicates a duration of 10 seconds, with 12 | 0.87 | 0.039 | 0.635 |
| 10054 | atrial fibrillation. st segments are depressed in i, ,avl, v | The ECG analysis reveals a duration of 10 seconds, with 14 Q | 1.01 | 0.078 | 0.625 |
| **MEAN** | | | **0.62** | **0.044** | **0.560** |

## ECG IMAGE branch (n=20, PTB-XL fold 10)

| ecg_id | reference (PTB-XL/RTLP) | generated | BLEU | ROUGE-1 | clin.cosine |
|---|---|---|---:|---:|---:|
| 9823 | sinusrytm vÄnster el-axel vÄnstersidigt skÄnkelblock ÖvervÄg | The ECG shows sinus rhythm with normal ST, T, and QRS comple | 0.00 | 0.000 | 0.283 |
| 2146 | sinus rhythm first degree av block, pr interval 0.24 seconds | The ECG shows sinus rhythm with normal QRS complexes, P wave | 1.38 | 0.085 | 0.661 |
| 15899 | sinus rhythm location type normal incomplete right bundle br |  | 0.00 | 0.000 | -0.014 |
| 17909 | sinus rhythm position type normal r-s # transition in v deri | The ECG shows a sinus rhythm with a normal QRS complex, P wa | 1.45 | 0.200 | 0.497 |
| 10180 | sinus rhythm. voltages are high in chest leads suggesting lv | The ECG shows a sinus rhythm with normal QRS complexes, P wa | 2.14 | 0.300 | 0.770 |
| 3092 | premature ventricular contraction(s). sinus rhythm. voltages | The ECG shows sinus rhythm with a consistent PR interval in  | 2.43 | 0.143 | 0.691 |
| 1745 | sinus rhythm position type normal atrial overload left possi | The ECG shows sinus rhythm with normal QRS complexes, P wave | 2.03 | 0.178 | 0.428 |
| 8083 | sinusrhythm location type normal moderate amplitude criteria | The ECG shows a normal sinus rhythm with normal QRS complexe | 0.74 | 0.039 | 0.370 |
| 15400 | sinus rhythm suspected p-sinistrocardiale location type norm | The ECG shows a sinus rhythm with regular P waves and a cons | 1.80 | 0.131 | 0.634 |
| 11664 | sinus arrhythmia p-sinistrocardiale location type normal qrs | The ECG shows a sinus rhythm with normal QRS complexes, P wa | 0.87 | 0.206 | 0.686 |
| 16905 | atrial fibrillation. st segments are depressed in v4,5,6. t  | The ECG shows sinus rhythm with normal ST and T wave segment | 3.17 | 0.300 | 0.782 |
| 3122 | sinus rhythm. q waves in ii, iii, avf consistent with old in | The ECG shows sinus rhythm with normal QRS complexes, P wave | 1.38 | 0.235 | 0.756 |
| 5975 | sinusrhythm position type normal normal ecg | The ECG shows sinus rhythm with a regular P wave and consist | 0.00 | 0.009 | 0.607 |
| 15548 | sinus bradycardia position type normal normal ekg | The ECG shows a sinus rhythm with normal QRS complexes and n | 0.42 | 0.058 | 0.585 |
| 2596 | sinus rhythm position type normal left posterior hemiblock | The ECG shows a sinus rhythm with normal QRS complexes, P wa | 2.09 | 0.143 | 0.503 |
| 19256 | sinus bradycardia position type normal normal ekg | The ECG shows sinus rhythm with normal QRS complexes, P wave | 0.66 | 0.090 | 0.608 |
| 3508 | sinus rhythm and junctional rhythm. non-specific t wave flat | The ECG shows a sinus rhythm with normal QRS complexes and n | 2.98 | 0.203 | 0.695 |
| 8533 | sinus rhythm location type normal path. repolarization proba | The ECG shows a regular sinus rhythm with a normal heart rat | 1.05 | 0.096 | 0.552 |
| 8450 | sinus rhythm. low limb lead voltage. st segments are depress | The ECG shows sinus rhythm with a normal QRS complex and P w | 2.83 | 0.175 | 0.736 |
| 10054 | atrial fibrillation. st segments are depressed in i, ,avl, v | The ECG shows a sinus rhythm with a consistent PR interval i | 0.69 | 0.035 | 0.621 |
| **MEAN** | | | **1.41** | **0.131** | **0.572** |

## Signal vs image (means)

| branch | BLEU | ROUGE-1 | clinical cosine |
|---|---:|---:|---:|
| ECG signal | 0.62 | 0.044 | 0.560 |
| ECG image | 1.41 | 0.131 | 0.572 |
