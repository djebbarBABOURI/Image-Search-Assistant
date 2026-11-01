# image-search-assistant

Serveur [MCP](https://modelcontextprotocol.io/) de classification d'images basé sur un **ResNet18** entraîné sur le jeu de données **Oxford-IIIT Pets** (races de chats et de chiens).

## Sommaire

- [Prérequis](#prérequis)
- [Installation](#installation)
- [Télécharger les images](#télécharger-les-images)
- [Construire le cache](#construire-le-cache)
- [Lancement](#lancement)
- [Outils, ressources et prompts MCP](#outils-ressources-et-prompts-mcp)
- [Chatbot : LLM, prompt et évaluation des appels d'outils](#chatbot--llm-prompt-et-évaluation-des-appels-doutils)
- [Entraînement et export du modèle spécialiste](#entraînement-et-export-du-modèle-spécialiste)
- [Évaluation des modèles et recherche](#évaluation-des-modèles-et-recherche)
- [Structure du projet](#structure-du-projet)

## Prérequis

- Python 3.10 ou plus récent
- [uv](https://docs.astral.sh/uv/)
- Le modèle entraîné : `models/model.pt` (voir [Checkpoints et exports fournis](#checkpoints-et-exports-fournis))
- Les images : `data/images/` (`.jpg`, `.jpeg`, `.png`)

## Installation

Aucune installation manuelle : `uv` crée un environnement temporaire à partir de `requirements.txt` au premier lancement.

> Le premier lancement télécharge `torch` et `torchvision` (plusieurs centaines de Mo). Les suivants utilisent le cache.

Le bloc d'évaluation fait exception : il n'a pas besoin de PyTorch (voir [Évaluation des modèles et recherche](#évaluation-des-modèles-et-recherche)).

## Télécharger les images

Les images viennent du jeu de données [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) (environ 750 Mo). Depuis la racine du projet (PowerShell) :

```powershell
New-Item -ItemType Directory -Force data | Out-Null
Invoke-WebRequest -Uri "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz" -OutFile data\images.tar.gz
tar -xzf data\images.tar.gz -C data
Remove-Item data\images.tar.gz
```

Les images se retrouvent dans `data/images/`.

## Construire le cache

Le serveur lit les prédictions dans `models/export.npz`. Ce fichier se génère une seule fois, depuis la racine du projet :

```powershell
uv run --with-requirements requirements.txt python -m src.training.export --run training_runs/7_sgd_100ep_final/finetune_seed2 --out data/export.npz
```


## Lancement

### Mode 1 : MCP Inspector (tester les outils)

```powershell
.\run.ps1
```

Ouvrez le lien `http://localhost:6274/?MCP_PROXY_AUTH_TOKEN=...` affiché dans le terminal (le token change à chaque démarrage). `Ctrl+C` arrête le serveur.

### Mode 2 : serveur HTTP

Le serveur écoute sur `http://localhost:8000/mcp`.

```powershell
.\run_server.ps1
```

avec `run_server.ps1` :

```powershell
Set-Location $PSScriptRoot
uv run --with-requirements requirements.txt python src/mcp_server/server.py
```

Le cache doit exister avant le premier lancement (voir [Construire le cache](#construire-le-cache)).

### Mode 3 : chatbot (client web)

Le client relie le serveur MCP à un LLM local via Ollama et sert une interface web. Ce mode nécessite le **Mode 2** : le serveur HTTP doit déjà tourner.

Il faut 3 processus, chacun dans son terminal :

1. **Ollama** : `ollama serve`
2. **Serveur MCP** : `.\run_server.ps1`
3. **Client** : `.\run_client.ps1`

avec `run_client.ps1` :

```powershell
Set-Location $PSScriptRoot
uv run --with-requirements requirements.txt --with fastapi --with uvicorn --with httpx python src/chatbot/client.py
```

Ouvrez ensuite http://localhost:8080.

## Outils, ressources et prompts MCP

### Outils

| Outil | Description |
|---|---|
| `classify(image_id, top_k)` | Prédit la race d'une image et renvoie les meilleures probabilités |
| `find_similar(image_id, k)` | Trouve les k images les plus proches dans l'espace des embeddings |
| `filter_images(label, min_confidence, limit)` | Liste les images prédites comme une race donnée |
| `count_by(label, min_confidence)` | Compte les images par race prédite |
| `top_confusions(label, k)` | Races les plus souvent confondues avec la race donnée |
| `list_breeds()` | Liste les 37 races et leur espèce |

### Ressources

| URI | Contenu |
|---|---|
| `pets://classes` | Les 37 races |
| `pets://stats` | Statistiques générales de la collection |
| `pets://metrics` | Accuracy globale du modèle |
| `pets://image/{image_id}` | Résumé d'une image |
| `pets://breed/{label}` | Résumé d'une race |

### Prompts

| Prompt | Arguments | Rôle |
|---|---|---|
| `identify_image` | `image_id` | Identifie la race d'une image et la fiabilité de la prédiction |
| `find_look_alikes` | `image_id` | Cherche les images proches et leurs races |
| `analyze_confusions` | `label` | Analyse les confusions du modèle sur une race |
| `audit_low_confidence` | `threshold` (optionnel) | Repère les races les plus incertaines |
| `breed_report` | `label` | Rapport sur une race : volume, qualité, confusions, exemples |

Les identifiants d'images sont les noms de fichier sans extension (ex. `Abyssinian_1`). La race « réelle » est déduite du nom de fichier.

> **Réglages tirés de l'évaluation.** La similarité cosinus donne de meilleurs résultats que la distance euclidienne (p@5 de 89,0 % contre 87,8 % pour le modèle fine-tuné). Un seuil `min_confidence = 0,8` garde environ 85 % des images pour une accuracy d'environ 95 %. Les fonctions `knn` et `l2_normalize` de `src/evaluation/retrieval.py` et `species_of` de `src/evaluation/analysis.py` peuvent être réutilisées côté serveur.

## Chatbot : LLM, prompt et évaluation des appels d'outils

### Choix du LLM

Le chatbot utilise un petit modèle local servi par [Ollama](https://ollama.com/), par défaut **`qwen2.5:3b`** : il gère l'appel d'outils (*function calling*) d'Ollama, tient sur un ordinateur portable et répond en quelques secondes sur CPU. Il se change sans toucher au code :

| Variable | Défaut | Rôle |
|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5:3b` | Modèle utilisé |
| `OLLAMA_HOST` | `http://localhost:11434` | Serveur Ollama (mettre l'adresse de l'API Ollama du CRIL pour l'utiliser) |
| `MCP_URL` | `http://localhost:8000/mcp` | Serveur MCP |

Le modèle se télécharge une fois avec `ollama pull qwen2.5:3b`. L'interface web permet aussi de choisir un autre modèle installé.

### Prompt système et boucle d'outils

Le prompt système (`src/chatbot/config.py`) décrit les outils, le format des identifiants d'images (`Abyssinian_1`) et des races (`British_Shorthair`, `staffordshire_bull_terrier`), et demande de réutiliser le résultat d'un outil comme argument du suivant. La boucle (`src/chatbot/agent.py`) envoie la conversation et la liste des outils MCP à Ollama, exécute les appels demandés sur le serveur MCP, renvoie les résultats au LLM, et recommence jusqu'à une réponse sans appel (6 étapes au plus). C'est ce qui permet les requêtes qui enchaînent plusieurs outils.

Côté usage, les races proches (bull terriers, Ragdoll et Birman, Basset Hound et Beagle) sont les plus souvent confondues : le modèle peut renvoyer un voisin de race voisine avec une similarité élevée.

### Prompts, ressources et endpoints du client
Endpoints exposés par le client (`src/chatbot/client.py`, port 8080) :

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/` | Interface web |
| GET | `/static/...` | Fichiers statiques (CSS, JS) |
| GET | `/api/tools` | Outils MCP visibles dans l'interface |
| GET | `/api/models` | Modèles Ollama installés et modèle par défaut |
| POST | `/api/chat` | Envoie la conversation à l'agent, renvoie la réponse et les appels d'outils |
| POST | `/api/upload` | Enregistre une image envoyée (10 Mo max) et renvoie son `image_id` |
| GET | `/api/image/{image_id}?size=thumb\|full` | Image de la collection ou téléversée |
| GET | `/api/prompts` | Prompts MCP et leurs arguments |
| POST | `/api/prompts/{nom}` | Exécute un prompt (`{"arguments": {...}}`), renvoie son texte et les ressources citées |
| GET | `/api/resources` | Ressources MCP et modèles de ressources |
| GET | `/api/resource?uri=pets://...` | Contenu d'une ressource |

### Jeu de requêtes de test

`tests/chatbot_queries.json` contient 20 requêtes avec l'outil et les arguments attendus : 13 à un seul outil, 6 qui en enchaînent deux et une qui ne doit appeler aucun outil. Par exemple :

```json
{"id": 15, "query": "Classe l'image Maine_Coon_30, puis compte combien d'images de la collection sont prédites comme cette race.",
 "expected": [{"tool": "classify", "args": {"image_id": "Maine_Coon_30"}}, {"tool": "count_by"}]}
```

Seuls les arguments écrits dans `args` sont vérifiés. Le second appel d'une chaîne n'a souvent pas d'arguments attendus, car ils dépendent du premier résultat. Un argument omis par le LLM est accepté s'il vaut la valeur par défaut de l'outil. Les races et identifiants sont comparés sans tenir compte de la casse, des espaces ou de l'extension. `"ordered": false` accepte les deux appels dans n'importe quel ordre.

### Lancer l'évaluation

Avec Ollama et le serveur MCP lancés (mode 3, étapes 1 et 2), depuis la racine :

```powershell
uv run --with-requirements requirements.txt python src/chatbot/eval_queries.py
uv run --with-requirements requirements.txt python src/chatbot/eval_queries.py --models qwen2.5:3b,llama3.2:3b
```

Options : `--models` (modèles séparés par des virgules), `--only 1,15,16` (sous-ensemble de requêtes), `--queries`, `--out-dir` (défaut `runs/chatbot/`). La température est fixée à 0 pour que les résultats soient reproductibles. Le script vérifie d'abord que les images citées existent dans la collection.

Il affiche chaque requête avec les appels faits, écrit le détail dans `runs/chatbot/<modèle>.json` et termine par un tableau :

| Mesure | Définition |
|---|---|
| Bon outil | Requêtes où chaque outil attendu a été appelé, sans regarder les arguments |
| Bons appels | Requêtes où chaque appel attendu a été fait avec les bons arguments (et dans l'ordre pour une chaîne) : **le taux de bons appels** |
| Rappel des appels | Part des appels attendus retrouvés, toutes requêtes confondues |
| Chaînes réussies | Taux de bons appels sur les requêtes à plusieurs outils |
| Appels en trop | Appels qui ne correspondent à aucun appel attendu |

Les tests de la logique de score ne demandent ni Ollama ni le serveur :

```powershell
python -m pytest tests/test_chatbot_scoring.py
```

## Entraînement et export du modèle spécialiste

Le dossier `src/training/` entraîne le modèle spécialiste (ResNet-18 de torchvision) sur Oxford-IIIT Pets, puis exporte ses prédictions et ses embeddings pour l'évaluation.

| Fichier | Rôle |
|---|---|
| `src/training/train.py` | Entraînement en mode `finetune` (poids ImageNet) ou `scratch` (poids aléatoires) |
| `src/training/export.py` | Export des probabilités et des embeddings de toutes les images dans un fichier `.npz` |
| `src/export_format.py` | Format commun des exports, lu par l'évaluation |
| `training_runs/` | Résultats (`results.json`) de toutes les expériences d'entraînement, avec un README qui les résume |

### Installation

L'entraînement a besoin de PyTorch et de torchvision. Depuis la racine du projet :

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install torch torchvision "numpy<2"
```

Testé avec Python 3.12, PyTorch 2.2.2 et torchvision 0.17.2, sur CPU (MacBook Pro Intel).
Sur un Mac Intel, PyTorch ne s'installe pas avec Python 3.13 ou 3.14 : utilisez Python 3.12 et `pip install torch==2.2.2 torchvision==0.17.2 "numpy<2"`.

### Données

Aucun téléchargement manuel : au premier lancement, `train.py` télécharge Oxford-IIIT Pets (environ 800 Mo) dans `data/oxford-iiit-pet/` grâce à `torchvision.datasets.OxfordIIITPet`.

- Le découpage officiel est utilisé : 3 680 images `trainval` et 3 669 images `test`.
- `trainval` est découpé en 85 % entraînement (3 128 images) et 15 % validation (552 images), avec une graine fixe (`SPLIT_SEED = 42`) : le découpage est identique pour tous les entraînements et tous les membres du groupe.
- La validation sert à choisir le meilleur modèle (meilleure accuracy top-1) et à décider de l'arrêt anticipé. Le test n'est évalué qu'une seule fois, à la fin, sur le modèle choisi.

### Reproduire le réglage final

Réglage final : SGD avec momentum 0,9, warm-up de 5 époques puis scheduler cosinus, weight decay 5e-4, 100 époques au maximum avec early stopping (patience 15), batch de 32 images en 224 × 224, augmentation de données (recadrage aléatoire, retournement horizontal, variations de couleur, rotation, effacement aléatoire).

```bash
# From scratch (learning rate de départ 0,025)
python -m src.training.train --mode scratch --seed 0 --epochs 100 --optimizer sgd --lr 0.025 --weight-decay 5e-4 --warmup 5 --patience 15 --out-dir training_runs/7_sgd_100ep_final

# Fine-tuning (learning rate de départ 0,0025)
python -m src.training.train --mode finetune --seed 0 --epochs 100 --optimizer sgd --lr 0.0025 --weight-decay 5e-4 --warmup 5 --patience 15 --out-dir training_runs/7_sgd_100ep_final
```

Les résultats ont été obtenus avec les seeds 0 à 4. Pour toutes les lancer à la suite :

```bash
# Linux et macOS
for s in 0 1 2 3 4; do
  python -m src.training.train --mode scratch  --seed $s --epochs 100 --optimizer sgd --lr 0.025  --weight-decay 5e-4 --warmup 5 --patience 15 --out-dir training_runs/7_sgd_100ep_final
  python -m src.training.train --mode finetune --seed $s --epochs 100 --optimizer sgd --lr 0.0025 --weight-decay 5e-4 --warmup 5 --patience 15 --out-dir training_runs/7_sgd_100ep_final
done
```

```powershell
# PowerShell
foreach ($s in 0..4) {
  python -m src.training.train --mode scratch  --seed $s --epochs 100 --optimizer sgd --lr 0.025  --weight-decay 5e-4 --warmup 5 --patience 15 --out-dir training_runs/7_sgd_100ep_final
  python -m src.training.train --mode finetune --seed $s --epochs 100 --optimizer sgd --lr 0.0025 --weight-decay 5e-4 --warmup 5 --patience 15 --out-dir training_runs/7_sgd_100ep_final
}
```

Durée sur CPU : environ 4 à 5 minutes par époque, soit environ 7 à 8 heures pour un entraînement from scratch et 2 à 4 heures pour un fine-tuning (arrêté plus tôt par l'early stopping). Sur macOS, `caffeinate -i` devant la commande empêche la mise en veille pendant l'entraînement.

Chaque entraînement écrit dans `<out-dir>/<mode>_seed<seed>/` :

| Fichier | Contenu |
|---|---|
| `model.pt` | Poids du meilleur modèle (meilleure accuracy de validation) |
| `results.json` | Réglages utilisés, courbes par époque (loss, accuracy, learning rate), époque du meilleur modèle, époque d'arrêt, accuracy top-1 et top-5 sur le test |

### Options de `train.py`

| Option | Défaut | Rôle |
|---|---|---|
| `--mode` | obligatoire | `finetune` (poids ImageNet) ou `scratch` (poids aléatoires) |
| `--seed` | 0 | Graine aléatoire (ordre des images, augmentations, initialisation) |
| `--epochs` | 50 | Nombre maximal d'époques |
| `--batch-size` | 32 | Taille des batchs |
| `--lr` | 1e-4 (finetune), 1e-3 (scratch) | Learning rate de départ |
| `--optimizer` | `adamw` | `adamw` ou `sgd` (SGD avec momentum 0,9) |
| `--weight-decay` | 0.01 | Weight decay |
| `--warmup` | 0 | Nombre d'époques de warm-up avant le scheduler cosinus |
| `--patience` | 10 | Early stopping : nombre d'époques sans amélioration de la validation (0 pour désactiver) |
| `--augment` | `standard` | `standard` (variations de couleur et rotation) ou `trivial` (TrivialAugmentWide) |
| `--label-smoothing` | 0.0 | Label smoothing de la loss |
| `--mixup` | désactivé | Active MixUp ou CutMix sur chaque batch |
| `--data-dir` | `data` | Dossier du dataset |
| `--out-dir` | `runs` | Dossier de sortie |

Les valeurs par défaut correspondent au réglage intermédiaire à 50 époques : le réglage final passe ses options explicitement (voir plus haut).

### Exporter un modèle pour l'évaluation

```bash
python -m src.training.export --run training_runs/7_sgd_100ep_final/finetune_seed2 --out data/export_finetune_sgd100_seed2.npz
```

L'export recharge le `model.pt` du dossier indiqué, calcule pour les 7 349 images les probabilités des 37 races et l'embedding (vecteur de 512 nombres pris avant la dernière couche), puis écrit un fichier `.npz` d'environ 14 Mo. Il recalcule aussi l'accuracy sur le test et vérifie qu'elle correspond à `test_top1` dans `results.json`.

Pour exporter les 10 modèles du réglage final, avec les noms attendus par `src/evaluation/run_all.py` :

```bash
# Linux et macOS
for m in finetune scratch; do
  for s in 0 1 2 3 4; do
    python -m src.training.export --run training_runs/7_sgd_100ep_final/${m}_seed$s --out data/export_${m}_sgd100_seed$s.npz
  done
done
```

```powershell
# PowerShell
foreach ($m in "finetune", "scratch") {
  foreach ($s in 0..4) {
    python -m src.training.export --run training_runs/7_sgd_100ep_final/${m}_seed$s --out data/export_${m}_sgd100_seed$s.npz
  }
}
```

Les chemins des images stockés dans ces exports commencent par `oxford-iiit-pet/images/` : pour l'évaluation avec images, utilisez `--data-dir data` (le dossier qui contient `oxford-iiit-pet/`).

### Checkpoints et exports fournis

Les poids des modèles (fichiers `model.pt` d'environ 43 Mo) et les exports (`.npz` d'environ 14 Mo) ne sont pas versionnés dans Git et ne peuvent pas être joints à l'archive (limite de 50 Mo par fichier sur Moodle). Ceux du réglage final sont disponibles sur Google Drive, en lecture seule :

https://drive.google.com/drive/folders/1r4YiyYaa79OIBZwgzJATQsJZXxvrLSwS?usp=sharing

| Dossier du Drive | Contenu | À copier dans |
|---|---|---|
| `7_sgd_100ep_final/` | les 10 modèles du réglage final (`<mode>_seed<k>/model.pt` et `results.json`) | `training_runs/` |
| `exports/` | les 10 exports `export_<mode>_sgd100_seed<k>.npz` | `data/` |

Avec les exports, les métriques se recalculent directement avec `src/evaluation/run_all.py`, sans PyTorch ni réentraînement (voir [Évaluation des modèles et recherche](#évaluation-des-modèles-et-recherche)).

Le serveur MCP utilise `models/model.pt`, qui est le meilleur modèle (fine-tuning seed 2, 90,1 % top-1 sur le test). Après avoir copié les checkpoints :

```bash
mkdir -p models
cp training_runs/7_sgd_100ep_final/finetune_seed2/model.pt models/model.pt
```

Les résultats des expériences intermédiaires (dossiers 1 à 6 de `training_runs/`) sont versionnés dans leurs fichiers `results.json`. Leurs checkpoints ne sont pas fournis.

### Résultats de l'entraînement (réglage final, 5 seeds, test)

| Mode | Top-1 | Top-5 |
|---|---|---|
| Fine-tuning | 89,2 ± 0,6 % | 99,0 ± 0,1 % |
| From scratch | 70,9 ± 0,8 % | 92,8 ± 0,5 % |

La progression de toutes les expériences (5, 30, 50 et 100 époques, tests de régularisation, changement d'optimiseur) est résumée dans `training_runs/README.md`.

## Évaluation des modèles et recherche

Le dossier `src/evaluation/` évalue les modèles entraînés sur Oxford-IIIT Pets. Il lit les exports `.npz` produits par `src/training/`, calcule les métriques, les baselines et les analyses d'erreurs, puis agrège les seeds (moyenne ± écart-type).

Il n'entraîne rien et n'a pas besoin de PyTorch : il travaille uniquement sur les probabilités et les embeddings déjà exportés.

### Installation

Python 3.10 ou plus récent, depuis la racine du dépôt :

```bash
pip install numpy matplotlib pillow
```

### Ce qu'il faut avoir avant de lancer

| Élément | Où | Détail |
|---|---|---|
| Exports des modèles | `data/export_<config>_seed<k>.npz` | Un fichier par configuration et par seed, par exemple `export_finetune_sgd100_seed2.npz` |
| Images | dossier `--data-dir` | Les chemins stockés dans les exports (`image_paths`) sont relatifs à `--data-dir` |

Les exports contiennent : `task`, `image_paths`, `class_names`, `probs`, `embeddings`, `labels`, `split`. Ils sont lus par `src.export_format.load_export`.

Le nom du fichier compte. Tout ce qui précède `_seed<k>` est le nom de la configuration, et les seeds d'une même configuration sont regroupées automatiquement dans les tableaux.

> **Attention au dossier des images.** Selon la façon dont les exports ont été produits, `image_paths` peut pointer vers `oxford-iiit-pet/images/...` (donc `data/oxford-iiit-pet/images/`) ou vers `images/...` (comme dans l'arborescence de ce dépôt, `data/images/`). Le dossier indiqué par `--data-dir` doit contenir le chemin exact stocké dans les exports. À défaut, relancez avec `--no-images`.

### Tout relancer en une commande

```bash
python -m src.evaluation.run_all --exports-dir data --data-dir data --runs-dir runs
```

Cette commande :

1. évalue chaque `export_<config>_seed<k>.npz` trouvé dans `data/` ;
2. écrit les résultats de chaque run dans `runs/<config>_seed<k>/` ;
3. génère les figures qualitatives (12 erreurs tirées au hasard, exemples de recherche) ;
4. agrège les seeds et écrit les tableaux dans `runs/`.

| Option | Défaut | Rôle |
|---|---|---|
| `--exports-dir` | `data` | Dossier des exports `.npz` |
| `--data-dir` | `data` | Racine des images (baseline pixels et figures) |
| `--runs-dir` | `runs` | Dossier de sortie |
| `--pixel-size` | `32` | Taille des images pour la baseline pixels bruts |
| `--n-errors` | `12` | Nombre d'erreurs tirées au hasard par figure |
| `--no-images` | désactivé | Sans baseline pixels ni figures (si les images sont absentes) |

La baseline pixels charge toutes les images la première fois, puis met le résultat en cache dans `data/pixels_32.npy`. Les lancements suivants sont plus rapides.

### Résultats produits

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

> `runs/chatbot/` est réservé à l'évaluation du chatbot (voir plus haut) et ne contient pas de run d'évaluation de modèle.

### Lancer une étape seule

Évaluer un seul export (sans `--data-dir`, la baseline pixels est ignorée) :

```bash
python -m src.evaluation.evaluate --export data/export_finetune_sgd100_seed2.npz --out runs/finetune_sgd100_seed2 --data-dir data
```

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

### Ajouter une configuration ou une ablation

Aucune modification de code n'est nécessaire. Il suffit d'exporter les nouveaux modèles sous la forme `data/export_<nouvelle_config>_seed<k>.npz` (3 à 5 seeds), puis de relancer `run_all`. La nouvelle configuration devient une colonne de `summary.md`.

Pour qu'une ablation soit valable, un seul facteur doit changer (même optimiseur, mêmes seeds, même nombre d'époques), et le choix du checkpoint doit se faire sur `val`.

### Métriques calculées

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

### Résultats de référence (5 seeds, test, n = 3 669)

| Métrique | finetune_sgd100 | scratch_sgd100 |
|---|---|---|
| Top-1 | 89,2 ± 0,6 % | 70,9 ± 0,8 % |
| Top-5 | 99,0 ± 0,1 % | 92,8 ± 0,5 % |
| F1 macro | 89,1 ± 0,6 % | 70,5 ± 0,9 % |
| ECE | 0,027 ± 0,006 | 0,109 ± 0,005 |
| p@5 (galerie test, cosinus) | 89,0 ± 0,3 % | 63,4 ± 1,2 % |
| p@5 pixels bruts | 5,8 % | 5,8 % |
| Classe majoritaire (top-1) | 2,4 % | 2,4 % |

### Problèmes fréquents

- **`Aucun fichier export_<config>_seed<k>.npz`** : vérifiez `--exports-dir` et le nom des fichiers.
- **Les images ne sont pas trouvées** : `image_paths` est relatif à `--data-dir` (voir l'avertissement plus haut). À défaut, relancez avec `--no-images`.
- **Caractères illisibles sous Windows** : les fichiers sont écrits en UTF-8. Ouvrez-les avec un éditeur qui lit l'UTF-8.
- **Baselines identiques pour toutes les configurations, avec un écart-type de 0,0** : c'est normal, elles ne dépendent pas du modèle.

## Structure du projet

```
image-search-assistant/
├── data/
│   ├── images/            images (non versionnées)
│   ├── oxford-iiit-pet/   dataset téléchargé par l'entraînement (non versionné)
│   ├── export_<config>_seed<k>.npz   exports des modèles (non versionnés)
│   └── pixels_32.npy      cache de la baseline pixels (généré)
├── models/                model.pt, export.npz.npz (non versionnés)
├── runs/                  résultats d'évaluation (runs/chatbot/ pour le chatbot)
├── training_runs/         résultats des entraînements (results.json), un dossier par expérience
├── src/
│   ├── search_index.py    chargement du modèle, images, prédiction
│   ├── export_format.py   format commun des exports .npz
│   ├── mcp_server/        serveur MCP
│   ├── chatbot/           client LLM, prompt, évaluation des appels d'outils
│   ├── evaluation/        métriques, recherche, baselines, agrégation
│   │   ├── metrics.py       matrice de confusion, précision, rappel, F1, top-k, ECE, paires confondues
│   │   ├── retrieval.py     similarités, k plus proches voisins, precision@k, galeries
│   │   ├── baselines.py     classe majoritaire, recherche aléatoire, pixels bruts
│   │   ├── analysis.py      analyse par espèce, courbe confiance et couverture
│   │   ├── evaluate.py      évaluation complète d'un export
│   │   ├── qualitative.py   figures d'erreurs aléatoires et d'exemples de recherche
│   │   ├── aggregate.py     moyenne ± écart-type sur les seeds, tableaux Markdown et JSON
│   │   └── run_all.py       orchestration : exports, figures, tableaux
│   └── training/          entraînement et export
│       ├── train.py         entraînement du ResNet-18 (finetune ou scratch)
│       └── export.py        export des probabilités et des embeddings (.npz)
├── tests/                 tests pytest et requêtes de test du chatbot
├── tools/                 scripts utilitaires
├── requirements.txt
├── run.ps1
├── run_client.ps1
└── run_server.ps1
```
