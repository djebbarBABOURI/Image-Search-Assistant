from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from src.export_format import load_export
from .retrieval import knn

def _show(ax, path, title, color="black"):
    from PIL import Image
    with Image.open(path) as im:
        ax.imshow(im.convert("RGB"))
    ax.set_title(title, fontsize=7, color=color)
    ax.axis("off")

def random_errors(ex, n=12, seed=0) -> list[dict]:
    test = np.where(ex.split == "test")[0]
    pred = ex.probs[test].argmax(1)
    wrong = test[pred != ex.labels[test]]
    rng = np.random.default_rng(seed)
    pick = rng.choice(wrong, size=min(n, len(wrong)), replace=False)
    rows = []
    for i in pick:
        order = np.argsort(-ex.probs[i])[:3]
        rows.append(dict(image_id=str(ex.image_paths[i]), true=str(ex.class_names[ex.labels[i]]),
                         pred=str(ex.class_names[order[0]]), confidence=float(ex.probs[i, order[0]]),
                         prob_true=float(ex.probs[i, ex.labels[i]]),
                         top3=[(str(ex.class_names[j]), float(ex.probs[i, j])) for j in order]))
    return rows

def plot_errors(rows, data_dir, path, ncols=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nrows = int(np.ceil(len(rows) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.2 * nrows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, r in zip(np.atleast_1d(axes).ravel(), rows):
        _show(ax, Path(data_dir) / r["image_id"],
              f"vrai: {r['true']}\nprédit: {r['pred']} ({r['confidence']:.0%})", "firebrick")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

def plot_retrieval(ex, data_dir, path, n_queries=4, k=5, seed=0, gallery="test", metric="cosine"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    test = np.where(ex.split == "test")[0]
    rng = np.random.default_rng(seed)
    q = rng.choice(test, size=n_queries, replace=False)
    gi = np.where(ex.split == "test")[0] if gallery == "test" else np.arange(len(ex.split))
    nb, sim = knn(ex.embeddings, q, gi, k, metric)
    fig, axes = plt.subplots(n_queries, k + 1, figsize=(2.6 * (k + 1), 2.9 * n_queries))
    for r in range(n_queries):
        _show(axes[r, 0], Path(data_dir) / ex.image_paths[q[r]],
              f"REQUÊTE\n{ex.class_names[ex.labels[q[r]]]}", "navy")
        for c in range(k):
            j = nb[r, c]
            ok = ex.labels[j] == ex.labels[q[r]]
            _show(axes[r, c + 1], Path(data_dir) / ex.image_paths[j],
                  f"{ex.class_names[ex.labels[j]]}\nsim={sim[r, c]:.2f}", "seagreen" if ok else "firebrick")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--export", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    ex = load_export(args.export)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = random_errors(ex, args.n, args.seed)
    (out / "errors.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_errors(rows, args.data_dir, out / "errors.png")
    plot_retrieval(ex, args.data_dir, out / "retrieval_examples.png", seed=args.seed)
    print(f"{len(rows)} erreurs aléatoires -> {out/'errors.png'} ; recherche -> {out/'retrieval_examples.png'}")

if __name__ == "__main__":
    main()
