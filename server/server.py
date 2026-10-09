from mcp.server.mcpserver import MCPServer
import logging
from PIL import Image
from torchvision import models, transforms

import torch
import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pets-server")

mcp = MCPServer(name="pets-server", version="1.0.0", description="Oxford Pets MCP Server")

BASE = Path(__file__).parent
MODEL_PATH = Path(os.getenv("PETS_MODEL", BASE / "model.pt"))
IMAGES_DIR = Path(os.getenv("PETS_IMAGES_DIR", BASE / "images"))
CLASSES_PATH = BASE / "classes.json"
EXTS = {".jpg", ".jpeg", ".png"}

DEFAULT_CLASSES = [
    "Abyssinian", "american_bulldog", "american_pit_bull_terrier", "basset_hound",
    "beagle", "Bengal", "Birman", "Bombay", "boxer", "British_Shorthair",
    "chihuahua", "Egyptian_Mau", "english_cocker_spaniel", "english_setter",
    "german_shorthaired", "great_pyrenees", "havanese", "japanese_chin",
    "keeshond", "leonberger", "Maine_Coon", "miniature_pinscher", "newfoundland",
    "Persian", "pomeranian", "pug", "Ragdoll", "Russian_Blue", "saint_bernard",
    "samoyed", "scottish_terrier", "shiba_inu", "Siamese", "Sphynx",
    "staffordshire_bull_terrier", "wheaten_terrier", "yorkshire_terrier",
]
CLASSES = json.loads(CLASSES_PATH.read_text()) if CLASSES_PATH.exists() else DEFAULT_CLASSES
 
TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
 
S: dict = {}
 
 
def species(name: str) -> str:
    return "cat" if name[0].isupper() else "dog"
 
 
def load_model() -> torch.nn.Module:
    if not MODEL_PATH.exists():
        raise SystemExit(f"Modèle introuvable : {MODEL_PATH} (variable PETS_MODEL)")
    m = models.resnet18(num_classes=len(CLASSES))
    m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    return m.eval()
 
 
def init() -> None:
    S["model"] = load_model()
    files = [p for p in IMAGES_DIR.rglob("*") if p.suffix.lower() in EXTS] if IMAGES_DIR.exists() else []
    S["files"] = {p.stem: p for p in files}
    log.info("Modèle chargé (%d classes), %d images dans %s", len(CLASSES), len(files), IMAGES_DIR)


@mcp.tool()
def classify(image_id: str, top_k: int = 3) -> dict:
    """Predict the breed of an image and give the top probabilities."""


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