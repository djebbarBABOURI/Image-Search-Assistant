"""Contrat d'export partagé (Khadija écrit, Yacine et Djebbar lisent).

Un fichier .npz contient, pour chaque image du jeu de données :

  task          : "single" (piste A, une classe par image) ou "multi" (piste B)
  image_paths   : (N,)   chemins relatifs des images, servent d'identifiant (image_id)
  class_names   : (C,)   noms des classes, dans l'ordre des colonnes de probs
  probs         : (N, C) probabilités prédites (softmax en A, sigmoïde en B)
  embeddings    : (N, D) vecteur avant la couche de classification
  labels        : (N,)   indice de la vraie classe (A)   ou   (N, C) multi-hot (B)
  split         : (N,)   "train", "val" ou "test"
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SPLITS = {"train", "val", "test"}


@dataclass
class Export:
    task: str
    image_paths: np.ndarray
    class_names: np.ndarray
    probs: np.ndarray
    embeddings: np.ndarray
    labels: np.ndarray
    split: np.ndarray

    def validate(self) -> None:
        n, c = self.probs.shape
        assert self.task in {"single", "multi"}, f"task invalide: {self.task}"
        assert self.image_paths.shape == (n,), "image_paths doit avoir la forme (N,)"
        assert self.class_names.shape == (c,), "class_names doit avoir la forme (C,)"
        assert self.embeddings.ndim == 2 and self.embeddings.shape[0] == n, "embeddings: (N, D)"
        assert self.split.shape == (n,) and set(self.split.tolist()) <= SPLITS, "split invalide"
        assert len(set(self.image_paths.tolist())) == n, "image_paths doit être unique"
        assert np.all((self.probs >= 0) & (self.probs <= 1)), "probs doit être dans [0, 1]"
        if self.task == "single":
            assert self.labels.shape == (n,), "labels (piste A): forme (N,)"
            assert np.allclose(self.probs.sum(axis=1), 1.0, atol=1e-3), "softmax: somme = 1"
        else:
            assert self.labels.shape == (n, c), "labels (piste B): forme (N, C) multi-hot"


def save_export(path: str | Path, ex: Export) -> None:
    ex.validate()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        task=np.array(ex.task),
        image_paths=ex.image_paths.astype(str),
        class_names=ex.class_names.astype(str),
        probs=ex.probs.astype(np.float32),
        embeddings=ex.embeddings.astype(np.float32),
        labels=ex.labels,
        split=ex.split.astype(str),
    )


def load_export(path: str | Path) -> Export:
    d = np.load(path, allow_pickle=False)
    ex = Export(
        task=str(d["task"]),
        image_paths=d["image_paths"],
        class_names=d["class_names"],
        probs=d["probs"],
        embeddings=d["embeddings"],
        labels=d["labels"],
        split=d["split"],
    )
    ex.validate()
    return ex
