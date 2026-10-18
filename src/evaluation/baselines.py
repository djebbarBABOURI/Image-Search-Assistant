from __future__ import annotations

from pathlib import Path

import numpy as np

from .metrics import confusion_matrix, per_class_prf
from .retrieval import evaluate_retrieval, gallery_indices, precision_at_k


def majority_baseline(ex, split: str = "test") -> dict:
    """Prédit toujours la classe la plus fréquente du train."""
    y_train = ex.labels[ex.split == "train"].astype(int)
    C = ex.probs.shape[1]
    maj = int(np.bincount(y_train, minlength=C).argmax())
    y = ex.labels[ex.split == split].astype(int)
    pred = np.full_like(y, maj)
    prf = per_class_prf(confusion_matrix(y, pred, C))
    return dict(majority_class=str(ex.class_names[maj]), top1=float((pred == y).mean()),
                macro_f1=float(prf["f1"].mean()))


def random_retrieval_baseline(ex, ks=(1, 5, 10), gallery: str = "test", seed: int = 0) -> dict:
    """Voisins tirés au hasard dans la galerie (plancher pour la precision@k)."""
    rng = np.random.default_rng(seed)
    q = np.where(ex.split == "test")[0]
    gi = gallery_indices(ex.split, gallery)
    nb = np.empty((len(q), max(ks)), dtype=int)
    for r, qi in enumerate(q):
        cand = gi[gi != qi]
        nb[r] = rng.choice(cand, size=max(ks), replace=False)
    return precision_at_k(ex.labels.astype(int), q, nb, ks)


def load_pixels(image_paths, data_dir, size: int = 32, cache: str | Path | None = None) -> np.ndarray:
    """Images réduites à size×size RGB et aplaties -> (N, 3*size*size) float32 dans [0, 1]."""
    if cache is not None and Path(cache).exists():
        arr = np.load(cache)
        if arr.shape[0] == len(image_paths):
            return arr
    from PIL import Image
    feats = []
    for p in image_paths:
        with Image.open(Path(data_dir) / str(p)) as im:
            feats.append(np.asarray(im.convert("RGB").resize((size, size)), dtype=np.float32).ravel() / 255.0)
    arr = np.stack(feats)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, arr)
    return arr


def pixel_retrieval_baseline(ex, data_dir, size: int = 32, ks=(1, 5, 10), cache=None) -> dict:
    pix = load_pixels(ex.image_paths, data_dir, size, cache)
    return evaluate_retrieval(pix, ex.labels.astype(int), ex.split, ks)
