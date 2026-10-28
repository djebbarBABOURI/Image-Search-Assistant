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
- [Structure du projet](#structure-du-projet)

## Prérequis

- Python 3.10 ou plus récent
- [uv](https://docs.astral.sh/uv/)
- Le modèle entraîné : `models/model.pt`
- Les images : `data/images/` (`.jpg`, `.jpeg`, `.png`)

## Installation

Aucune installation manuelle : `uv` crée un environnement temporaire à partir de `requirements.txt` au premier lancement.

> Le premier lancement télécharge `torch` et `torchvision` (plusieurs centaines de Mo). Les suivants utilisent le cache.

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

Le serveur lit les prédictions dans `models/pets_cache.npz`. Ce fichier se génère une seule fois, depuis la racine du projet :

```powershell
uv run --with-requirements requirements.txt python tools/build_cache.py
```

Le calcul prend quelques minutes sur CPU. Relancez cette commande si vous changez de modèle ou si le contenu de `data/images/` change (supprimez d'abord `models/pets_cache.npz`).

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

`find_look_alikes`, `identify_image`, `analyze_confusions`, `audit_low_confidence`, `breed_report`.

 Les identifiants d'images sont les noms de fichier sans extension (ex. `Abyssinian_1`). La race « réelle » est déduite du nom de fichier.

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

## Structure du projet

```
image-search-assistant/
├── data/images/           images (non versionnées)
├── models/                model.pt, pets_cache.npz (non versionnés)
├── runs/                  résultats d'évaluation (runs/chatbot/ pour le chatbot)
├── src/
│   ├── search_index.py    chargement du modèle, images, prédiction
│   ├── mcp_server/        serveur MCP
│   ├── chatbot/           client LLM, prompt, évaluation des appels d'outils
│   ├── evaluation/        métriques
│   └── training/          entraînement et export
├── tests/                 tests pytest et requêtes de test du chatbot
├── tools/                 scripts utilitaires
├── requirements.txt
├── run.ps1
├── run_client.ps1
└── run_server.ps1