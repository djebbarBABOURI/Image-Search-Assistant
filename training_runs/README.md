# Expériences d'entraînement (ResNet-18, Oxford-IIIT Pets)

Chaque dossier correspond à une expérience, dans l'ordre chronologique.
Chaque sous-dossier `<mode>_seed<seed>/` contient le `results.json` d'un entraînement
(réglages, courbes par époque, résultats sur le test).
Les modèles (`model.pt`) et les exports (`.npz`) sont sur le Drive du projet.

| Dossier | Réglage | Seeds | Fine-tuning top-1 | Scratch top-1 |
|---|---|---|---|---|
| 1_adamw_5ep | 5 époques, AdamW, lr constant, augmentation simple | 3 | 88,0 ± 0,5 % | 16,8 ± 0,6 % |
| 2_adamw_30ep | 30 époques, AdamW, lr constant | 1 (scratch) | — | 41,5 % |
| 3_adamw_50ep | 50 époques, AdamW, cosine, augmentation renforcée, early stopping | 5 | 88,2 ± 0,8 % | 57,8 ± 0,8 % |
| 4_testA_labelsmoothing_trivialaugment | 3 + label smoothing 0,1 + TrivialAugment | 1 (scratch) | — | 56,9 % |
| 5_testB_mixup_cutmix | 4 + MixUp/CutMix | 1 (scratch) | — | 44,4 % |
| 6_sgd_50ep | 50 époques, SGD + momentum 0,9 + warm-up | 1 (scratch) | — | 63,4 % |
| 7_sgd_100ep_final | **Réglage final** : 100 époques, SGD + momentum 0,9 + warm-up | 5 | **89,2 ± 0,6 %** | **70,9 ± 0,8 %** |

Moyennes ± écarts-types calculés à partir des valeurs arrondies des results.json.
L'évaluation complète du réglage final se trouve dans `runs/`.

Commande du réglage final (scratch ; pour le fine-tuning : `--mode finetune --lr 0.0025`) :

    python -m src.training.train --mode scratch --seed 0 --epochs 100 --optimizer sgd --lr 0.025 --weight-decay 5e-4 --warmup 5 --patience 15 --out-dir training_runs/7_sgd_100ep_final
