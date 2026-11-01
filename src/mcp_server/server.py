import sys
import logging
import os
from pathlib import Path
import base64
import io
import uuid
import re
from PIL import Image
from mcp.server.mcpserver import MCPServer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.search_index import load_classes, load_model, list_images, predict, probs_for, species, probs_batch, embs_and_probs_batch

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pets-server")

mcp = MCPServer(name="pets-server", version="1.0.0", description="Oxford Pets MCP Server")

MODEL_PATH = Path(os.getenv("PETS_MODEL", ROOT / "models" / "model.pt"))
IMAGES_DIR = Path(os.getenv("PETS_IMAGES_DIR", ROOT / "data" / "images"))
UPLOADS_DIR = Path(os.getenv("PETS_UPLOADS_DIR", ROOT / "data" / "uploads"))
EXTS = {".jpg", ".jpeg", ".png"}
CLASSES_PATH = ROOT / "models" / "classes.json"

MODELS_DIR = ROOT / "models"
EXPORT_CANDIDATES = [
    "export.npz",
    *[f"export_{mode}_sgd100_seed{s}.npz" for mode in ("finetune", "scratch") for s in range(5)],
    "pets_cache.npz",
]

# Mettre un nom ici pour forcer un export précis (sinon None = détection automatique)
EXPORT_FORCED: str | None = None
def find_export() -> Path:
    for name in ([EXPORT_FORCED] if EXPORT_FORCED else EXPORT_CANDIDATES):
        p = MODELS_DIR / name
        if p.exists():
            return p
    return MODELS_DIR / "export.npz" 


EXPORT_PATH = find_export()

CLASSES: list[str] = []      # rempli dans init() depuis l'export
CLASS_POS: dict[str, int] = {}
SPECIES: dict[str, str] = {}

S: dict = {}


def norm(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


def species(name: str) -> str:
    return SPECIES.get(name, "cat" if name[0].isupper() else "dog")


def class_of(label: str):
    return CLASS_POS.get(norm(label))


def true_label(image_id: str):
    return class_of(image_id.rsplit("_", 1)[0])


def build_cache() -> None:
    model = load_model(MODEL_PATH, len(CLASSES))
    files = list_images(IMAGES_DIR)
    ids = sorted(files)
    log.info("Calcul des prédictions pour %d images...", len(ids))
    embs, probs = embs_and_probs_batch(
        model, [files[i] for i in ids],
        on_progress=lambda n, t: log.info("%d / %d", n, t) if n % 320 == 0 or n == t else None)
    np.savez(CACHE_PATH, ids=np.array(ids), probs=probs, embs=embs)
    log.info("Cache écrit : %s", CACHE_PATH)

def top(probs, k: int) -> list[dict]:
    k = max(1, min(k, len(CLASSES)))
    return [{"label": CLASSES[j], "species": species(CLASSES[j]),
             "probability": round(float(probs[j]), 4)}
            for j in np.argsort(-probs)[:k]]

def vec(image_id: str):
    """Retourne les probabilités et l'embedding d'une image."""
    stem = Path(image_id).stem

    # Image du dataset
    i = S["pos"].get(stem)
    if i is not None:
        return stem, S["probs"][i], S["embs"][i], i

    # Image téléversée
    if stem not in S["up"]:
        path = next(
            (UPLOADS_DIR / f"{stem}{ext}"
             for ext in EXTS
             if (UPLOADS_DIR / f"{stem}{ext}").is_file()),
            None
        )

        if path is None:
            return None

        embs, probs = embs_and_probs_batch(S["model"], [path])
        emb = embs[0]
        norm = np.linalg.norm(emb)

        if norm == 0:
            return None

        S["up"][stem] = (
            probs[0],
            emb / norm
        )

    probs, emb = S["up"][stem]
    return stem, probs, emb, None
def pick(d, *names, required=True):
    for n in names:
        if n in d.files:
            return d[n]
    if required:
        raise KeyError(f"Aucune des clés {names} dans {EXPORT_PATH}. Clés présentes : {d.files}")
    return None


def init() -> None:
    if not EXPORT_PATH.exists():
        raise RuntimeError(
            f"Export absent : {EXPORT_PATH}. Lancez d'abord "
            "uv run --with-requirements requirements.txt python -m src.training.export --run training_runs/7_sgd_100ep_final/finetune_seed2 --out models/export.npz")
    d = np.load(EXPORT_PATH, allow_pickle=False)

    # Classes : export, sinon classes.json
    raw_classes = pick(d, "class_names", "classes", required=False)
    classes = [str(c) for c in raw_classes] if raw_classes is not None else load_classes(CLASSES_PATH)
    CLASSES[:] = classes
    CLASS_POS.clear()
    CLASS_POS.update({norm(c): i for i, c in enumerate(classes)})

    ids = [Path(str(p)).stem for p in pick(d, "image_paths", "paths", "ids")]
    labels = pick(d, "labels", "label", "y", required=False)
    probs = pick(d, "probs", "probabilities")
    emb = pick(d, "embeddings", "embs", "emb").astype(np.float32)
    split = pick(d, "split", "splits", required=False)

    # Espèce déduite du nom de fichier (majuscule = chat)
    SPECIES.clear()
    for stem in ids:
        c = class_of(stem.rsplit("_", 1)[0])
        if c is not None:
            SPECIES[classes[c]] = "cat" if stem[0].isupper() else "dog"

    S.update(
        model=load_model(MODEL_PATH, len(classes)),
        files=list_images(IMAGES_DIR),
        ids=ids,
        probs=probs,
        labels=labels,
        split=split,
        pos={s: i for i, s in enumerate(ids)},
        embs=emb / np.linalg.norm(emb, axis=1, keepdims=True),
    )
    S["up"] = {}
    log.info("Export chargé (%d classes), %d images", len(classes), len(ids))

def ensure_init() -> None:
    if not S:
        init()


@mcp.tool()
def register_image(filename: str, data_base64: str) -> dict:
    """Enregistre une image envoyée par le client et retourne son identifiant."""
    try:
        raw = base64.b64decode(data_base64, validate=True)
        with Image.open(io.BytesIO(raw)) as source:
            img = source.convert("RGB")
    except Exception:
        return {"error": "Image invalide"}

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    image_id = f"upload_{uuid.uuid4().hex[:8]}"
    img.save(UPLOADS_DIR / f"{image_id}.jpg", format="JPEG", quality=95)

    return {
        "image_id": image_id,
        "filename": Path(filename).name
    }

@mcp.tool()
def get_image(image_id: str, size: str = "thumb") -> dict:
    """Retourne une image en Base64 pour le client web."""
    ensure_init()

    if size not in ("thumb", "full"):
        return {"error": "Taille invalide"}

    if not image_id or not re.fullmatch(r"[\w. -]+", image_id):
        return {"error": "Identifiant invalide"}

    stem = Path(image_id).stem

    # Recherche dans le dataset
    image_path = S["files"].get(stem)

    # Recherche dans les images téléversées
    if image_path is None:
        image_path = next(
            (UPLOADS_DIR / f"{stem}{ext}"
             for ext in EXTS
             if (UPLOADS_DIR / f"{stem}{ext}").is_file()),
            None
        )

    if image_path is None or not Path(image_path).is_file():
        return {"error": f"Image introuvable : {image_id}"}

    try:
        with Image.open(image_path) as source:
            img = source.copy()
            original_format = source.format or "JPEG"

        if size == "thumb":
            img.thumbnail((320, 320))
            img = img.convert("RGB")
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            content = buffer.getvalue()
            mime = "image/jpeg"
        else:
            content = Path(image_path).read_bytes()
            mime = Image.MIME.get(
                original_format,
                "application/octet-stream"
            )

        return {
            "image_id": stem,
            "mime": mime,
            "data_base64": base64.b64encode(content).decode("ascii")
        }

    except Exception:
        log.exception("Erreur lors de la lecture de %s", image_id)
        return {"error": "Impossible de lire l'image"}

@mcp.tool()
def classify(image_id: str, top_k: int = 3) -> dict:
    """Prédit la race d'une image du dataset ou téléversée."""
    ensure_init()

    result = vec(image_id)
    if result is None:
        return {"error": f"Image inconnue : {image_id}"}

    iid, probs, _, _ = result

    return {
        "image_id": iid,
        "predictions": top(probs, top_k)
    }

@mcp.tool()
def find_similar(image_id: str, k: int = 5) -> dict:
    """Find the k images closest to a given image in the embedding space."""
    ensure_init()
    v = vec(image_id)
    if v is None:
        return {"error": f"Image inconnue : {image_id}"}
    iid, _, emb, i = v
    sims = S["embs"] @ emb
    if i is not None:
        sims[i] = -np.inf
    out = []
    for j in np.argsort(-sims)[:max(1, k)]:
        best = top(S["probs"][j], 1)[0]
        out.append({"image_id": S["ids"][j], "similarity": round(float(sims[j]), 4),
                    "predicted_breed": best["label"], "confidence": best["probability"]})
    return {"image_id": iid, "neighbors": out}

  

@mcp.tool()
def filter_images(label: str, min_confidence: float = 0.0, limit: int = 10) -> dict:
    """List the images PREDICTED as a given breed with a minimum confidence."""
    ensure_init()
    c = class_of(label)
    if c is None:
        return {"error": f"Race inconnue : {label}. Voir la ressource pets://classes"}
    pred, conf = S["probs"].argmax(1), S["probs"].max(1)
    hits = np.where((pred == c) & (conf >= min_confidence))[0]
    hits = hits[np.argsort(-conf[hits])]
    return {"label": CLASSES[c], "total_matching": int(len(hits)),
            "images": [{"image_id": S["ids"][j], "confidence": round(float(conf[j]), 4)}
                       for j in hits[:max(1, limit)]]}


@mcp.tool()
def count_by(label: str = "", min_confidence: float = 0.0) -> dict:
    """Count images by predicted breed."""
    ensure_init()
    pred, conf = S["probs"].argmax(1), S["probs"].max(1)
    keep = conf >= min_confidence
    counts = np.bincount(pred[keep], minlength=len(CLASSES))
    if label:
        c = class_of(label)
        if c is None:
            return {"error": f"Race inconnue : {label}. Voir la ressource pets://classes"}
        return {"label": CLASSES[c], "count": int(counts[c]), "min_confidence": min_confidence}
    order = np.argsort(-counts)
    return {"min_confidence": min_confidence, "total": int(keep.sum()),
            "counts": {CLASSES[j]: int(counts[j]) for j in order if counts[j] > 0}}


@mcp.tool()
def top_confusions(label: str, k: int = 3) -> dict:
    """For a given breed, list the breeds the model most often predicts instead (confusions), with example image ids."""
    ensure_init()
    c = class_of(label)
    if c is None:
        return {"error": f"Race inconnue : {label}. Voir la ressource pets://classes"}
    pred = S["probs"].argmax(1)
    conf = S["probs"].max(1)
    members = [i for i, s in enumerate(S["ids"])
           if true_label(s) == c]
    if not members:
        return {"error": f"Aucune image étiquetée {CLASSES[c]} (nom de fichier) dans la collection"}
    wrong = [i for i in members if pred[i] != c]
    by_class: dict[int, list[int]] = {}
    for i in wrong:
        by_class.setdefault(int(pred[i]), []).append(i)
    ranked = sorted(by_class.items(), key=lambda kv: -len(kv[1]))[:max(1, k)]
    return {
        "label": CLASSES[c],
        "total_images": len(members),
        "misclassified": len(wrong),
        "error_rate": round(len(wrong) / len(members), 4),
        "confused_with": [
            {"predicted_breed": CLASSES[j], "count": len(img_idx),
            "examples": [{"image_id": S["ids"][k_], "confidence": round(float(conf[k_]), 4)}
                        for k_ in sorted(img_idx, key=lambda n: -conf[n])[:3]]}
            for j, img_idx in ranked
        ],
    }

@mcp.tool()
def list_breeds() -> dict:
    """List the 37 valid breed names with their species (cat or dog)."""
    return {"count": len(CLASSES),
            "breeds": [{"name": c, "species": species(c)} for c in CLASSES]}


@mcp.resource("pets://classes")
def classes_resource() -> str:
    """Expose the 37 valid breed names, one per line, with their species."""
    return "\n".join(f"{c} ({species(c)})" for c in CLASSES)


@mcp.resource("pets://stats")
def stats_resource() -> str:
    """Expose general statistics about the collection and the predictions."""
    ensure_init()
    pred, conf = S["probs"].argmax(1), S["probs"].max(1)
    cats = sum(species(CLASSES[p]) == "cat" for p in pred)
    labeled = sum(true_label(i) is not None for i in S["ids"])
    return "\n".join([
        f"Images : {len(S['ids'])}",
        f"Images avec race connue (nom de fichier) : {labeled}",
        f"Prédites chat : {cats} | chien : {len(pred) - cats}",
        f"Confiance moyenne : {conf.mean():.3f}",
        f"Prédictions < 0.5 de confiance : {(conf < 0.5).sum()}",
    ])


@mcp.resource("pets://metrics")
def metrics_resource() -> str:
    """Expose the evaluation metrics of the trained model."""
    ensure_init()
    pred = S["probs"].argmax(1)
    pairs = [(i, true_label(s)) for i, s in enumerate(S["ids"])]
    pairs = [(i, t) for i, t in pairs if t is not None]
    idx = np.array([i for i, _ in pairs])
    truth = np.array([t for _, t in pairs])
    acc = float((pred[idx] == truth).mean())
    return f"Accuracy globale : {acc:.3f} sur {len(idx)} images étiquetées"
    

@mcp.resource("pets://image/{image_id}")
def image_resource(image_id: str) -> str:
    """Expose the summary of one image of the collection."""
    ensure_init()
    stem = Path(image_id).stem
    i = S["pos"].get(stem)
    if i is None:
        return f"Image inconnue : {image_id}"
    t = true_label(stem)
    preds = top(S["probs"][i], 3)
    lines = [
        f"Image : {stem}",
        f"Fichier : {S['files'].get(stem)}",
        f"Race (nom de fichier) : {CLASSES[t] if t is not None else 'inconnue'}",
        f"Race prédite : {preds[0]['label']} ({preds[0]['species']}) "
        f"avec une confiance de {preds[0]['probability']:.3f}",
    ]
    if t is not None:
        lines.append("Prédiction correcte : " + ("oui" if CLASSES[t] == preds[0]["label"] else "non"))
    lines.append("Top 3 : " + ", ".join(f"{p['label']} ({p['probability']:.3f})" for p in preds))
    return "\n".join(lines)

@mcp.resource("pets://breed/{label}")
def breed_resource(label: str) -> str:
    """Expose a summary of one breed: predictions count, accuracy, mean confidence."""
    ensure_init()
    c = class_of(label)
    if c is None:
        return f"Race inconnue : {label}. Voir pets://classes"
    pred, conf = S["probs"].argmax(1), S["probs"].max(1)
    predicted = pred == c
    members = np.array([i for i, s in enumerate(S["ids"]) if true_label(s) == c], dtype=int)
    lines = [
        f"Race : {CLASSES[c]} ({species(CLASSES[c])})",
        f"Images prédites comme {CLASSES[c]} : {int(predicted.sum())}",
        f"Confiance moyenne sur ces prédictions : "
        f"{(conf[predicted].mean() if predicted.any() else 0):.3f}",
        f"Images étiquetées {CLASSES[c]} : {len(members)}",
    ]
    if len(members):
        lines.append(f"Accuracy sur cette race : {float((pred[members] == c).mean()):.3f}")
    return "\n".join(lines)
 
@mcp.prompt()
def find_look_alikes(image_id: str) -> str:
    """Build a prompt asking for the neighbours of an image and their breeds."""
    return (
        f"Trouve les 5 images les plus proches de {image_id} avec find_similar. "
        f"Puis appelle classify sur {image_id} et sur chaque voisin. "
        "Présente un tableau (image, race prédite, confiance) et dis si les "
        "voisins ont la même race que l'image de départ."
    )

@mcp.prompt()
def identify_image(image_id: str) -> str:
    """Build a prompt to identify an image and explain how reliable the prediction is."""
    return (
        f"Appelle classify sur {image_id} avec top_k=5, et lis la ressource "
        f"pets://image/{image_id}. Dis quelle est la race la plus probable, "
        "si c'est un chat ou un chien, et si la prédiction est fiable "
        "(confiance, écart avec la 2e race). Si la confiance est faible, "
        "utilise find_similar pour appuyer ton analyse."
    )
 
 
@mcp.prompt()
def analyze_confusions(label: str) -> str:
    """Build a prompt to analyse the confusions of the model on one breed."""
    return (
        f"Appelle top_confusions pour la race {label} avec k=3. "
        "Pour chaque race confondue, explique pourquoi les deux races peuvent "
        "se ressembler visuellement. Consulte pets://breed/" + label +
        " pour comparer avec la précision globale, puis propose des pistes "
        "pour améliorer le modèle sur cette race."
    )
 
 
@mcp.prompt()
def audit_low_confidence(threshold: float = 0.5) -> str:
    """Build a prompt to audit the predictions with low confidence."""
    return (
        "Lis pets://stats et pets://metrics. Puis utilise count_by avec "
        f"min_confidence={threshold} et compare avec count_by sans seuil pour "
        "voir quelles races perdent le plus d'images. Résume les races les "
        "plus incertaines et recommande un seuil de confiance adapté."
    )
 
 
@mcp.prompt()
def breed_report(label: str) -> str:
    """Build a prompt to produce a full report about one breed."""
    return (
        f"Fais un rapport sur la race {label} : lis pets://breed/{label}, "
        f"appelle filter_images(label='{label}', min_confidence=0.9, limit=5) "
        f"pour les exemples les plus sûrs, count_by(label='{label}') et "
        f"top_confusions(label='{label}'). Présente le résultat en sections : "
        "volume, qualité, confusions, exemples."
    )

if __name__ == "__main__":
    init()
    mcp.run(transport="streamable-http")