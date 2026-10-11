# ECG report evaluation - results interpretation

## Reference subset (RTLP English translations - language detection no longer used)
- PTB-XL records with an RTLP translation available: 16247
- After validated_by_human filter: 12279
- Administrative placeholders excluded: 239
- Total reference pool available: 12040
- Records actually evaluated in this run: 300 (299 scored, 1 failed)

## Aggregate scores (mean +/- std, over scored records)
- BLEU: 4.29 +/- 5.60
- ROUGE-1 (F1): 0.262 +/- 0.207
- ROUGE-2 (F1): 0.135 +/- 0.201
- ROUGE-L (F1): 0.260 +/- 0.208
- BERTScore F1: 0.821 +/- 0.049

## Per-class breakdown (stratified sample)
- CD (n=60): BLEU 2.93, ROUGE-1 0.205, BERTScore F1 0.822
- HYP (n=60): BLEU 2.93, ROUGE-1 0.250, BERTScore F1 0.819
- MI (n=59): BLEU 1.74, ROUGE-1 0.159, BERTScore F1 0.793
- NORM (n=60): BLEU 11.14, ROUGE-1 0.497, BERTScore F1 0.873
- STTC (n=60): BLEU 2.65, ROUGE-1 0.200, BERTScore F1 0.798

## BERTScore contrastive baseline (mismatched pairs)
- Real pairs: 0.821 +/- 0.049
- Mismatched pairs (each generated text scored against a DIFFERENT record's reference): 0.819 +/- 0.049
- Gap (real - mismatched): +0.002
Raw BERTScore (no baseline rescaling) is known to compress into a narrow high band regardless of actual semantic match, so the real-pairs mean alone is not sufficient evidence of quality. A meaningfully positive gap here supports that the real pairs carry genuine signal above this model's floor; a small or negative gap would mean the 0.70-ish BERTScore F1 number is not distinguishable from noise and should not be cited as evidence of report quality on its own.

## Interpretation caveat
These scores measure textual/semantic overlap with a stylistically different, abbreviated reference - not clinical correctness. PTB-XL's reference reports are short, telegraphic clinical shorthand (e.g. "sinusrhythmus normales ekg"), while the generated text is a synthesized 3-5 sentence narrative. This length/style mismatch structurally depresses BLEU and ROUGE (both sensitive to exact n-gram/length overlap) even when the generated report is clinically consistent with the reference. A low BLEU/ROUGE score does not, by itself, mean the generated report is wrong. BERTScore is a semantic similarity measure and is comparatively less sensitive to this surface-form mismatch - weight it more heavily than BLEU/ROUGE when judging report quality from these numbers, but see the contrastive baseline above before treating the absolute BERTScore number as meaningful on its own. The evaluation sample is stratified to give roughly equal representation to each diagnostic class (see stratified_sample) rather than mirroring the reference pool's natural class distribution - the aggregate above is a stratified estimate of per-class performance, not a naturalistic average.

Additional caveat - machine-translated references: reference reports now come from RTLP's machine translation of PTB-XL's mostly-German original reports (language detection was found unreliable on this dataset - 13.3% false-positive rate - and is no longer used to build the reference pool; see cross_check_native_vs_rtlp and the module docstring). This introduces translation-quality noise as a further source of mismatch alongside the length/style and descriptive-vs-diagnostic gaps above - a generated text scored against an imperfectly translated reference may be penalized (or credited) for reasons unrelated to its own quality.