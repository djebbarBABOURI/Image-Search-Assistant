from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a, dtype=np.float64)
    np.divide(a, b, out=out, where=b > 0)
    return out


def per_class_prf(cm: np.ndarray) -> dict[str, np.ndarray]:
    """Calcul des métriques par classe à partir de la matrice de confusion."""
    tp = np.diag(cm).astype(np.float64)
    precision = _safe_div(tp, cm.sum(axis=0))   # TP / nb de prédictions de la classe
    recall = _safe_div(tp, cm.sum(axis=1))      # TP / nb d'images réelles de la classe
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return dict(precision=precision, recall=recall, f1=f1, support=cm.sum(axis=1))


def topk_accuracy(probs: np.ndarray, y_true: np.ndarray, k: int) -> float:
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float((topk == y_true[:, None]).any(axis=1).mean())


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """Calcul de l'ECE (Expected Calibration Error) selon https://arxiv.org/abs/1706.04599."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def most_confused_pairs(cm: np.ndarray, class_names, n: int = 10) -> list[dict]:
    """Retourne les paires de classes les plus confondues."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(-off, axis=None)[:n]
    rows = []
    for f in flat:
        i, j = np.unravel_index(f, off.shape)
        if off[i, j] == 0:
            break
        rows.append(dict(true=str(class_names[i]), pred=str(class_names[j]), count=int(off[i, j]),
                         rate=float(off[i, j] / max(cm[i].sum(), 1))))
    return rows


def top_confusions(cm: np.ndarray, class_names, target: str, n: int = 5) -> list[dict]:
    """Retourne les classes les plus confondues avec une classe donnée."""
    names = [str(c) for c in class_names]
    i = names.index(target)
    row = cm[i].copy()
    row[i] = 0
    order = np.argsort(-row)[:n]
    return [dict(pred=names[j], count=int(row[j]), rate=float(row[j] / max(cm[i].sum(), 1)))
            for j in order if row[j] > 0]


def classification_report(ex, split: str = "test", n_pairs: int = 10) -> dict:
    m = ex.split == split
    probs, y = ex.probs[m], ex.labels[m].astype(int)
    C = probs.shape[1]
    pred = probs.argmax(1)
    cm = confusion_matrix(y, pred, C)
    prf = per_class_prf(cm)
    names = [str(c) for c in ex.class_names]
    return dict(
        split=split, n=int(m.sum()),
        top1=float((pred == y).mean()), top5=topk_accuracy(probs, y, min(5, C)),
        macro_precision=float(prf["precision"].mean()), macro_recall=float(prf["recall"].mean()),
        macro_f1=float(prf["f1"].mean()), ece=expected_calibration_error(probs, y),
        mean_confidence=float(probs.max(1).mean()),
        per_class={names[c]: dict(precision=float(prf["precision"][c]), recall=float(prf["recall"][c]),
                                  f1=float(prf["f1"][c]), support=int(prf["support"][c])) for c in range(C)},
        most_confused=most_confused_pairs(cm, names, n_pairs),
        confusion_matrix=cm,
    )
