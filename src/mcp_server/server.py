import sys
import logging
import os
from pathlib import Path

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
CLASSES_PATH = ROOT / "models" / "classes.json"

CLASSES = load_classes(CLASSES_PATH)
CLASS_POS = {c.lower(): i for i, c in enumerate(CLASSES)}

S: dict = {}

def class_of(label: str):
    return CLASS_POS.get(label.strip().lower().replace(" ", "_"))


def true_label(image_id: str):
    return class_of(image_id.rsplit("_", 1)[0])

CACHE_PATH = ROOT / "models" / "pets_cache.npz"


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
    """Retourne (id, probs, embedding normalisé, index collection) ou None
    si l'image n'est pas dans la collection."""
    stem = Path(image_id).stem
    i = S["pos"].get(stem)
    if i is None:
        return None
    return stem, S["probs"][i], S["embs"][i], i

def init() -> None:
    if not CACHE_PATH.exists():
        raise RuntimeError("Cache absent : lancez d'abord  uv run --with-requirements requirements.txt python tools/build_cache.py")
    d = np.load(CACHE_PATH)
    ids = [str(i) for i in d["ids"]]
    emb = d["embs"]
    S.update(
        model=load_model(MODEL_PATH, len(CLASSES)),
        files=list_images(IMAGES_DIR),
        ids=ids,
        probs=d["probs"],
        pos={s: i for i, s in enumerate(ids)},
        embs=emb / np.linalg.norm(emb, axis=1, keepdims=True),
    )
    log.info("Modèle chargé (%d classes), %d images dans %s", len(CLASSES), len(ids), IMAGES_DIR)

def ensure_init() -> None:
    if not S:
        init()

@mcp.tool()
def classify(image_id: str, top_k: int = 3) -> dict:
    """Predict the breed of an image and give the top probabilities."""
    log.info("classify appelé avec %s", image_id)
    ensure_init()
    path = S["files"].get(Path(image_id).stem)
    if path is None:
        return {"error": f"Image inconnue : {image_id}"}
    return {"image_id": path.stem, "predictions": predict(S["model"], path, CLASSES, top_k)}


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
    

@mcp.resource("pets://image/{image_id}")
def image_resource(image_id: str) -> str:
    """Expose the summary of one image of the collection."""

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


if __name__ == "__main__":
    init()
    mcp.run()