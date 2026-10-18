# image-search-assistant

Serveur [MCP](https://modelcontextprotocol.io/) de classification d'images basé sur un **ResNet18** entraîné sur le jeu de données **Oxford-IIIT Pets** (races de chats et de chiens).

## Sommaire

- [Prérequis](#prérequis)
- [Installation](#installation)
- [Télécharger les images](#télécharger-les-images)
- [Construire le cache](#construire-le-cache)
- [Lancement](#lancement)
- [Outils, ressources et prompts MCP](#outils-ressources-et-prompts-mcp)
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

## Structure du projet

```
image-search-assistant/
├── data/images/           images (non versionnées)
├── models/                model.pt, pets_cache.npz (non versionnés)
├── src/
│   ├── search_index.py    chargement du modèle, images, prédiction
│   ├── mcp_server/        serveur MCP
│   ├── chatbot/           client
│   ├── evaluation/        métriques
│   └── training/          entraînement et export
├── tests/                 tests pytest
├── tools/                 scripts utilitaires
├── requirements.txt
└── run.ps1
├── run_client.ps1
└── run_server.ps1