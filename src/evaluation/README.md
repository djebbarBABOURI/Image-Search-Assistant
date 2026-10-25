# Évaluation et recherche (bloc 2)

Ce dossier (`src/evaluation/`) évalue les modèles entraînés sur Oxford-IIIT Pets (piste A). Il lit les exports `.npz` produits par le bloc « Données et entraînement », calcule les métriques, les baselines, les analyses d'erreurs, puis agrège les seeds (moyenne ± écart-type).

Il n'entraîne rien et n'a pas besoin de PyTorch : il travaille uniquement sur les probabilités et les embeddings déjà exportés.

## Installation

Python 3.10 ou plus récent, depuis la racine du dépôt :

```bash
pip install numpy matplotlib pillow
```

## Ce qu'il faut avoir avant de lancer

| Élément | Où | Détail |
|---|---|---|
| Exports des modèles | `data/export_<config>_seed<k>.npz` | Un fichier par configuration et par seed, par exemple `export_finetune_sgd100_seed2.npz` |
| Images | `data/oxford-iiit-pet/images/` | Les chemins stockés dans les exports (`image_paths`) sont relatifs à `--data-dir` |

Le format des exports est celui du sujet : `task`, `image_paths`, `class_names`, `probs`, `embeddings`, `labels`, `split`. Il est lu par `src.export_format.load_export`.

Le nom du fichier compte. Tout ce qui précède `_seed<k>` est le nom de la configuration, et les seeds d'une même configuration sont regroupées automatiquement dans les tableaux.

## Tout relancer en une commande

```bash
python -m src.evaluation.run_all --exports-dir data --data-dir data --runs-dir runs
```

Cette commande :

1. évalue chaque `export_<config>_seed<k>.npz` trouvé dans `data/` ;
2. écrit les résultats de chaque run dans `runs/<config>_seed<k>/` ;
3. génère les figures qualitatives (12 erreurs tirées au hasard, exemples de recherche) ;
4. agrège les seeds et écrit les tableaux dans `runs/`.

Options :

| Option | Défaut | Rôle |
|---|---|---|
| `--exports-dir` | `data` | Dossier des exports `.npz` |
| `--data-dir` | `data` | Racine des images (baseline pixels et figures) |
| `--runs-dir` | `runs` | Dossier de sortie |
| `--pixel-size` | `32` | Taille des images pour la baseline pixels bruts |
| `--n-errors` | `12` | Nombre d'erreurs tirées au hasard par figure |
| `--no-images` | désactivé | Sans baseline pixels ni figures (si les images sont absentes) |

La baseline pixels charge toutes les images la première fois, puis met le résultat en cache dans `data/pixels_32.npy`. Les lancements suivants sont plus rapides.

## Résultats produits

Pour chaque run, dans `runs/<config>_seed<k>/` :

| Fichier | Contenu |
|---|---|
| `eval.json` | Toutes les métriques (classification, recherche, baselines, analyses) |
| `confusion_matrix.npy` et `.png` | Matrice de confusion sur le test, normalisée par ligne pour l'image |
| `confidence_curve.png` | Couverture et accuracy selon le seuil de confiance |
| `errors.json` et `errors.png` | Erreurs tirées au hasard, avec top-3 et confiance |
| `retrieval_examples.png` | Quatre requêtes et leurs 5 plus proches voisins |

Pour l'ensemble des runs, dans `runs/` :

| Fichier | Contenu |
|---|---|
| `summary.md` | Tableau des métriques principales, moyenne ± écart-type par configuration |
| `summary_all.md` | Même tableau avec toutes les métriques |
| `summary.json` | Les mêmes valeurs en JSON (moyenne, écart-type, nombre de seeds) |

## Lancer une étape seule

Évaluer un seul export :

```bash
python -m src.evaluation.evaluate --export data/export_finetune_sgd100_seed2.npz --out runs/finetune_sgd100_seed2 --data-dir data
```

Sans `--data-dir`, la baseline pixels est ignorée.

Régénérer les figures qualitatives avec un autre tirage (écrire dans un autre dossier pour ne rien écraser) :

```bash
python -m src.evaluation.qualitative --export data/export_finetune_sgd100_seed2.npz --data-dir data --out runs/finetune_sgd100_seed2_alt8 --seed 8
```

Réagréger des `eval.json` existants :

```bash
# Linux et macOS
python -m src.evaluation.aggregate runs/*/eval.json --out runs/summary.md --all

# PowerShell
python -m src.evaluation.aggregate (Get-ChildItem runs\*\eval.json) --out runs/summary.md --all
```

Sans `--all`, seules les métriques principales sont écrites. L'option `--out` écrit aussi un `.json` à côté.

## Ajouter une configuration ou une ablation

Aucune modification de code n'est nécessaire. Il suffit d'exporter les nouveaux modèles sous la forme `data/export_<nouvelle_config>_seed<k>.npz` (3 à 5 seeds), puis de relancer `run_all`. La nouvelle configuration devient une colonne de `summary.md`.

Pour qu'une ablation soit valable, un seul facteur doit changer (même optimiseur, mêmes seeds, même nombre d'époques), et le choix du checkpoint doit se faire sur `val`.

## Métriques calculées

Toutes les métriques sont calculées sur `split == "test"`.

**Classification** : top-1, top-5, précision, rappel et F1 macro, ECE (15 intervalles), confiance moyenne, matrice de confusion, paires les plus confondues, métriques par classe.

**Recherche** : precision@k pour k = 1, 5 et 10, avec similarité cosinus ou distance euclidienne, et trois galeries (`test`, `trainval`, `all`). La requête est toujours exclue de ses voisins.

**Baselines** (identiques pour toutes les configurations, donc d'écart-type 0,0) :
- classe majoritaire (la plus fréquente du train) ;
- voisins tirés au hasard dans la galerie ;
- voisins sur les pixels bruts (images réduites à 32×32 RGB).

**Analyses** :
- `analysis.species.*` : part des erreurs qui restent dans la même espèce (chat ou chien) ;
- `analysis.confidence.t50` à `t95` : pour chaque seuil, nombre d'images, couverture et accuracy.

Dans `summary.md`, les clés ont la forme `retrieval.embedding.test.cosine.p@5` (type de représentation, galerie, métrique, k).

## Résultats de référence (5 seeds, test, n = 3 669)

| Métrique | finetune_sgd100 | scratch_sgd100 |
|---|---|---|
| Top-1 | 89,2 ± 0,6 % | 70,9 ± 0,8 % |
| Top-5 | 99,0 ± 0,1 % | 92,8 ± 0,5 % |
| F1 macro | 89,1 ± 0,6 % | 70,5 ± 0,9 % |
| ECE | 0,027 ± 0,006 | 0,109 ± 0,005 |
| p@5 (galerie test, cosinus) | 89,0 ± 0,3 % | 63,4 ± 1,2 % |
| p@5 pixels bruts | 5,8 % | 5,8 % |
| Classe majoritaire (top-1) | 2,4 % | 2,4 % |

## Pour les autres blocs

- **Serveur MCP** : la similarité cosinus donne de meilleurs résultats que la distance euclidienne (par exemple p@5 de 89,0 % contre 87,8 % pour le fine-tuné). Un seuil `min_confidence = 0,8` garde environ 85 % des images pour une accuracy d'environ 95 %. Les fonctions `knn` et `l2_normalize` de `retrieval.py` et `species_of` de `analysis.py` peuvent être réutilisées.
- **Chatbot** : les races proches (bull terriers, Ragdoll et Birman, Basset Hound et Beagle) sont les plus souvent confondues. Le modèle peut renvoyer un voisin de race voisine avec une similarité élevée.

## Structure

| Fichier | Rôle |
|---|---|
| `metrics.py` | Matrice de confusion, précision, rappel, F1, top-k, ECE, paires confondues |
| `retrieval.py` | Similarités, k plus proches voisins, precision@k, galeries |
| `baselines.py` | Classe majoritaire, recherche aléatoire, pixels bruts |
| `analysis.py` | Analyse par espèce, courbe confiance et couverture |
| `evaluate.py` | Évaluation complète d'un export |
| `qualitative.py` | Figures d'erreurs aléatoires et d'exemples de recherche |
| `aggregate.py` | Moyenne ± écart-type sur les seeds, tableaux Markdown et JSON |
| `run_all.py` | Orchestration : tous les exports, toutes les figures, tous les tableaux |

## Problèmes fréquents

- **`Aucun fichier export_<config>_seed<k>.npz`** : vérifie `--exports-dir` et le nom des fichiers.
- **Les images ne sont pas trouvées** : `image_paths` est relatif à `--data-dir`. Le dossier `oxford-iiit-pet/images/` doit se trouver dans ce dossier. À défaut, relance avec `--no-images`.
- **Caractères illisibles sous Windows** : les fichiers sont écrits en UTF-8. Ouvre-les avec un éditeur qui lit l'UTF-8.
- **Baselines identiques pour toutes les configurations, avec un écart-type de 0,0** : c'est normal, elles ne dépendent pas du modèle.
