import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
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

TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def probs_for(model, path):
    with torch.no_grad():
        x = TF(Image.open(path).convert("RGB")).unsqueeze(0)
        return torch.softmax(model(x), dim=1)[0].numpy()

def load_classes(classes_path: Path) -> list[str]:
    return json.loads(classes_path.read_text()) if classes_path.exists() else DEFAULT_CLASSES


def species(name: str) -> str:
    return "cat" if name[0].isupper() else "dog"


def load_model(model_path: Path, n_classes: int) -> torch.nn.Module:
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_path} (variable PETS_MODEL)")
    m = models.resnet18(num_classes=n_classes)
    m.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    return m.eval()


def list_images(images_dir: Path) -> dict[str, Path]:
    files = [p for p in images_dir.rglob("*") if p.suffix.lower() in EXTS] if images_dir.exists() else []
    return {p.stem: p for p in files}


def predict(model: torch.nn.Module, path: Path, classes: list[str], top_k: int = 3) -> list[dict]:
    x = TF(Image.open(path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x)[0], dim=0)
    p, idx = probs.topk(max(1, min(top_k, len(classes))))
    return [
        {"label": classes[i], "species": species(classes[i]), "probability": round(float(v), 4)}
        for v, i in zip(p, idx)
    ]

def probs_batch(model, paths, batch_size=32, on_progress=None):
    out = []
    for i in range(0, len(paths), batch_size):
        batch = torch.stack([TF(Image.open(p).convert("RGB")) for p in paths[i:i + batch_size]])
        with torch.no_grad():
            out.append(torch.softmax(model(batch), dim=1).numpy())
        if on_progress:
            on_progress(min(i + batch_size, len(paths)), len(paths))
    return np.concatenate(out)

def embs_and_probs_batch(model, paths, batch_size=32, on_progress=None):
    """Retourne (embeddings 512-d, probabilités) pour une liste d'images (resnet18)."""
    feats = {}
    hook = model.avgpool.register_forward_hook(
        lambda m, i, o: feats.__setitem__("x", o.flatten(1).numpy())
    )
    embs, probs = [], []
    try:
        for i in range(0, len(paths), batch_size):
            batch = torch.stack([TF(Image.open(p).convert("RGB")) for p in paths[i:i + batch_size]])
            with torch.no_grad():
                logits = model(batch)
            probs.append(torch.softmax(logits, dim=1).numpy())
            embs.append(feats["x"].copy())
            if on_progress:
                on_progress(min(i + batch_size, len(paths)), len(paths))
    finally:
        hook.remove()
    return np.concatenate(embs), np.concatenate(probs)