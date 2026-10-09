import sys
import logging
import os
from pathlib import Path

from mcp.server.mcpserver import MCPServer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.search_index import load_classes, load_model, list_images, predict, probs_for, species, probs_batch

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
    probs = probs_batch(model, [files[i] for i in ids],
                        on_progress=lambda n, t: log.info("%d / %d", n, t) if n % 320 == 0 or n == t else None)
    np.savez(CACHE_PATH, ids=np.array(ids), probs=probs)
    log.info("Cache écrit : %s", CACHE_PATH)


def init() -> None:
    if not CACHE_PATH.exists():
        raise RuntimeError("Cache absent : lancez d'abord  uv run --with-requirements requirements.txt python tools/build_cache.py")
    d = np.load(CACHE_PATH)
    S["model"] = load_model(MODEL_PATH, len(CLASSES))
    S["files"] = list_images(IMAGES_DIR)
    S["ids"] = list(d["ids"])
    S["probs"] = d["probs"]
    log.info("Modèle chargé (%d classes), %d images dans %s", len(CLASSES), len(S["ids"]), IMAGES_DIR)

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