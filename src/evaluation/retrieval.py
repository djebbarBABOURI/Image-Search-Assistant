from __future__ import annotations

import numpy as np

GALLERIES = ("test", "trainval", "all")


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def similarity(q: np.ndarray, g: np.ndarray, metric: str) -> np.ndarray:
    if metric == "cosine":
        return l2_normalize(q) @ l2_normalize(g).T
    if metric == "euclid":
        q2 = (q ** 2).sum(1)[:, None]
        g2 = (g ** 2).sum(1)[None, :]
        return -(q2 + g2 - 2 * q @ g.T)
    raise ValueError(f"metric inconnue: {metric}")


def gallery_indices(split: np.ndarray, gallery: str) -> np.ndarray:
    if gallery == "test":
        return np.where(split == "test")[0]
    if gallery == "trainval":
        return np.where(split != "test")[0]
    if gallery == "all":
        return np.arange(len(split))
    raise ValueError(f"galerie inconnue: {gallery}")


def knn(features: np.ndarray, query_idx: np.ndarray, gallery_idx: np.ndarray,
        k: int, metric: str = "cosine", chunk: int = 1024):
    all_idx, all_sim = [], []
    g = features[gallery_idx]
    for s in range(0, len(query_idx), chunk):
        qi = query_idx[s:s + chunk]
        sim = similarity(features[qi], g, metric)
        sim[qi[:, None] == gallery_idx[None, :]] = -np.inf
        top = np.argpartition(-sim, kth=min(k, sim.shape[1] - 1), axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sim, top, 1), axis=1)
        top = np.take_along_axis(top, order, 1)
        all_idx.append(gallery_idx[top])
        all_sim.append(np.take_along_axis(sim, top, 1))
    return np.concatenate(all_idx), np.concatenate(all_sim)