# ECG report evaluation - results interpretation

## Reference subset (RTLP English translations - language detection no longer used)
- PTB-XL records with an RTLP translation available: 16247
- After validated_by_human filter: 12279
- Administrative placeholders excluded: 239
- Total reference pool available: 12040
- Records actually evaluated in this run: 30 (30 scored, 0 failed)

## Aggregate scores (mean +/- std, over scored records)
- BLEU: 0.80 +/- 0.55
- ROUGE-1 (F1): 0.055 +/- 0.040
- ROUGE-2 (F1): 0.005 +/- 0.012
- ROUGE-L (F1): 0.044 +/- 0.032
- BERTScore F1: 0.708 +/- 0.028

## Per-class breakdown (stratified sample)
- CD (n=6): BLEU 0.74, ROUGE-1 0.042, BERTScore F1 0.696
- HYP (n=6): BLEU 0.69, ROUGE-1 0.040, BERTScore F1 0.702
- MI (n=6): BLEU 0.72, ROUGE-1 0.051, BERTScore F1 0.709
- NORM (n=6): BLEU 0.68, ROUGE-1 0.053, BERTScore F1 0.714
- STTC (n=6): BLEU 1.15, ROUGE-1 0.091, BERTScore F1 0.721

## BERTScore contrastive baseline (mismatched pairs)
- Real pairs: 0.708 +/- 0.028
- Mismatched pairs (each generated text scored against a DIFFERENT record's reference): 0.711 +/- 0.029
- Gap (real - mismatched): -0.003
Raw BERTScore (no baseline rescaling) is known to compress into a narrow high band regardless of actual semantic match, so the real-pairs mean alone is not sufficient evidence of quality. A meaningfully positive gap here supports that the real pairs carry genuine signal above this model's floor; a small or negative gap would mean the 0.70-ish BERTScore F1 number is not distinguishable from noise and should not be cited as evidence of report quality on its own.

## Interpretation caveat
These scores measure textual/semantic overlap with a stylistically different, abbreviated reference - not clinical correctness. PTB-XL's reference reports are short, telegraphic clinical shorthand (e.g. "sinusrhythmus normales ekg"), while the generated text is a synthesized 3-5 sentence narrative. This length/style mismatch structurally depresses BLEU and ROUGE (both sensitive to exact n-gram/length overlap) even when the generated report is clinically consistent with the reference. A low BLEU/ROUGE score does not, by itself, mean the generated report is wrong. BERTScore is a semantic similarity measure and is comparatively less sensitive to this surface-form mismatch - weight it more heavily than BLEU/ROUGE when judging report quality from these numbers, but see the contrastive baseline above before treating the absolute BERTScore number as meaningful on its own. The evaluation sample is stratified to give roughly equal representation to each diagnostic class (see stratified_sample) rather than mirroring the reference pool's natural class distribution - the aggregate above is a stratified estimate of per-class performance, not a naturalistic average.

Additional caveat - machine-translated references: reference reports now come from RTLP's machine translation of PTB-XL's mostly-German original reports (language detection was found unreliable on this dataset - 13.3% false-positive rate - and is no longer used to build the reference pool; see cross_check_native_vs_rtlp and the module docstring). This introduces translation-quality noise as a further source of mismatch alongside the length/style and descriptive-vs-diagnostic gaps above - a generated text scored against an imperfectly translated reference may be penalized (or credited) for reasons unrelated to its own quality.