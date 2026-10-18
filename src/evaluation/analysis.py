from __future__ import annotations
from pathlib import Path
import numpy as np

CAT_BREEDS = {"abyssinian", "bengal", "birman", "bombay", "british shorthair", "egyptian mau",
              "maine coon", "persian", "ragdoll", "russian blue", "siamese", "sphynx"}

DEFAULT_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)

def species_of(class_name: str) -> str:
    return "cat" if str(class_name).lower().replace("_", " ").strip() in CAT_BREEDS else "dog"

def species_analysis(ex, split: str = "test") -> dict:
    names = [str(c) for c in ex.class_names]
    sp = np.array([species_of(n) for n in names])
    if not (sp == "cat").any():
        return {}
    m = ex.split == split
    y = ex.labels[m].astype(int)
    pred = ex.probs[m].argmax(1)
    wrong = pred != y
    n_err = int(wrong.sum())
    same = int(((sp[y] == sp[pred]) & wrong).sum())
    return dict(
        species_accuracy=float((sp[y] == sp[pred]).mean()),
        n_errors=n_err,
        same_species_errors=same,
        cross_species_errors=n_err - same,
        same_species_error_share=float(same / max(n_err, 1)),
    )

def confidence_curve(ex, split: str = "test", thresholds=DEFAULT_THRESHOLDS) -> dict:
    m = ex.split == split
    probs, y = ex.probs[m], ex.labels[m].astype(int)
    conf, pred = probs.max(1), probs.argmax(1)
    out = {}
    for t in thresholds:
        sel = conf > t
        n = int(sel.sum())
        out[f"t{int(round(t * 100))}"] = dict(
            n=n, coverage=float(n / len(y)),
            accuracy=float((pred[sel] == y[sel]).mean()) if n else 0.0)
    return out

def plot_confidence_curve(ex, path, split: str = "test") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ts = np.linspace(0.0, 0.99, 100)
    cur = confidence_curve(ex, split, ts)
    cov = [v["coverage"] for v in cur.values()]
    acc = [v["accuracy"] if v["n"] else np.nan for v in cur.values()]
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot(ts, cov, label="couverture (part des images)")
    ax.plot(ts, acc, label="accuracy sur ces images")
    ax.set_xlabel("seuil de confiance t (confiance > t)")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(path), dpi=150)
    plt.close(fig)