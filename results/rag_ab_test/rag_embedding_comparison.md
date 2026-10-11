# RAG embedding-model comparison (current production index)

General `all-MiniLM-L6-v2` vs clinical `NeuML/pubmedbert-base-embeddings`, on the shipped 3-source / 83-chunk index. 14 clinical queries; each scored against its correct chunk + 2 hard negatives (other queries' answers) + 3 easy negatives (technical/meta chunks).


| model | Pearson r | top-1 accuracy | mean sim (pos) | mean sim (neg) | separation |
|---|---:|---:|---:|---:|---:|
| general  (all-MiniLM-L6-v2) | 0.770 | 13/14 (93%) | 0.646 | 0.233 | 0.412 |
| clinical (NeuML/pubmedbert) | 0.770 | 14/14 (100%) | 0.748 | 0.279 | 0.469 |
