import sys
import logging
import os
from pathlib import Path

from mcp.server.mcpserver import MCPServer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.search_index import load_classes, load_model, list_images, predict, species  # noqa: E402

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pets-server")

mcp = MCPServer(name="pets-server", version="1.0.0", description="Oxford Pets MCP Server")

MODEL_PATH = Path(os.getenv("PETS_MODEL", ROOT / "models" / "model.pt"))
IMAGES_DIR = Path(os.getenv("PETS_IMAGES_DIR", ROOT / "data" / "images"))
CLASSES_PATH = ROOT / "models" / "classes.json"

CLASSES = load_classes(CLASSES_PATH)

S: dict = {}


def init() -> None:
    S["model"] = load_model(MODEL_PATH, len(CLASSES))
    S["files"] = list_images(IMAGES_DIR)
    log.info("Modèle chargé (%d classes), %d images dans %s", len(CLASSES), len(S["files"]), IMAGES_DIR)


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


@mcp.tool()
def count_by(label: str = "", min_confidence: float = 0.0) -> dict:
    """Count images by predicted breed."""


@mcp.tool()
def list_breeds() -> dict:
    """List the 37 valid breed names with their species (cat or dog)."""
    return {"count": len(CLASSES),
            "breeds": [{"name": c, "species": species(c)} for c in CLASSES]}


@mcp.resource("pets://classes")
def classes_resource() -> str:
    """Expose the 37 valid breed names, one per line, with their species."""


@mcp.resource("pets://stats")
def stats_resource() -> str:
    """Expose general statistics about the collection and the predictions."""


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