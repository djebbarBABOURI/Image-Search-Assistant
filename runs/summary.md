| Métrique | finetune_sgd100 (n=5) | scratch_sgd100 (n=5) |
|---|---|---|
| classification.top1 | 89.2 ± 0.6 % | 70.9 ± 0.8 % |
| classification.top5 | 99.0 ± 0.1 % | 92.8 ± 0.5 % |
| classification.macro_f1 | 89.1 ± 0.6 % | 70.5 ± 0.9 % |
| classification.ece | 0.027 ± 0.006 | 0.109 ± 0.005 |
| retrieval.embedding.test.cosine.p@1 | 90.9 ± 0.2 % | 67.1 ± 1.0 % |
| retrieval.embedding.test.cosine.p@5 | 89.0 ± 0.3 % | 63.4 ± 1.2 % |
| retrieval.embedding.test.cosine.p@10 | 87.9 ± 0.2 % | 61.1 ± 1.1 % |
| retrieval.embedding.test.euclid.p@5 | 87.8 ± 0.3 % | 59.8 ± 1.2 % |
| retrieval.embedding.trainval.cosine.p@5 | 88.5 ± 0.2 % | 67.8 ± 1.0 % |
| retrieval.pixels.test.cosine.p@5 | 5.8 ± 0.0 % | 5.8 ± 0.0 % |
| baselines.majority.top1 | 2.4 ± 0.0 % | 2.4 ± 0.0 % |
| baselines.random_retrieval.p@5 | 2.7 ± 0.0 % | 2.7 ± 0.0 % |
| analysis.species.species_accuracy | 99.1 ± 0.1 % | 95.0 ± 0.2 % |
| analysis.species.same_species_error_share | 91.9 ± 0.9 % | 82.9 ± 0.8 % |
| analysis.confidence.t80.coverage | 84.6 ± 1.0 % | 65.0 ± 0.8 % |
| analysis.confidence.t80.accuracy | 95.4 ± 0.4 % | 87.3 ± 0.4 % |
