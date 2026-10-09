# image-search-assistant

Serveur [MCP](https://modelcontextprotocol.io/) de classification d'images basé sur un **ResNet18** entraîné sur le jeu de données **Oxford-IIIT Pets** (races de chats et de chiens).

## Sommaire

- [Prérequis](#prérequis)
- [Installation](#installation)
- [Télécharger les images](#télécharger-les-images)
- [Lancement](#lancement)
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

## Lancement

Depuis la racine du projet :

```powershell
.\run.ps1
```

Équivalent à :

```powershell
uv run --with-requirements requirements.txt mcp dev src/mcp_server/server.py
```

Pour lancer le serveur sans l'inspecteur :

```powershell
uv run --with-requirements requirements.txt python -m src.mcp_server.server
```

Ouvrez ensuite dans le navigateur le lien `http://localhost:6274/?MCP_PROXY_AUTH_TOKEN=...` affiché dans le terminal. Le token change à chaque démarrage. `Ctrl+C` arrête le serveur.


## Structure du projet

```
image-search-assistant/
├── data/images/           images (non versionnées)
├── models/                model.pt (non versionné), classes.json
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