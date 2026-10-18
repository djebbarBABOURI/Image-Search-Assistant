from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from src.export_format import load_export
from .analysis import confidence_curve, plot_confidence_curve, species_analysis
from .baselines import majority_baseline, pixel_retrieval_baseline, random_retrieval_baseline
from .metrics import classification_report
from .retrieval import evaluate_retrieval


def plot_confusion(cm: np.ndarray, class_names, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(class_names)), class_names, fontsize=7)
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Vraie classe")
    ax.set_title("Matrice de confusion (test, normalisée par ligne)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def evaluate_run(export_path, out_dir, data_dir=None, ks=(1, 5, 10), pixel_size=32) -> dict:
    ex = load_export(export_path)
    if ex.task != "single":
        raise SystemExit("Ce script couvre la piste A (task='single'). Piste B : F1/mAP à ajouter.")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    y = ex.labels.astype(int)

    report = classification_report(ex, "test")
    cm = report.pop("confusion_matrix")
    np.save(out / "confusion_matrix.npy", cm)
    plot_confusion(cm, ex.class_names, out / "confusion_matrix.png")
    plot_confidence_curve(ex, out / "confidence_curve.png")

    result = dict(
        name=out.name, export=str(export_path), n_test=report["n"], embedding_dim=int(ex.embeddings.shape[1]),
        classification=report,
        baselines=dict(majority=majority_baseline(ex),
                       random_retrieval=random_retrieval_baseline(ex, ks)),
        retrieval=dict(embedding=evaluate_retrieval(ex.embeddings, y, ex.split, ks)),
        analysis=dict(species=species_analysis(ex), confidence=confidence_curve(ex)),
    )

    if data_dir is not None:
        result["retrieval"]["pixels"] = pixel_retrieval_baseline(
            ex, data_dir, pixel_size, ks,
            cache=Path(data_dir) / f"pixels_{pixel_size}.npy")

    (out / "eval.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--export", required=True)
    p.add_argument("--out", required=True, help="ex. runs/finetune_seed0")
    p.add_argument("--data-dir", default=None, help="si fourni : baseline pixels bruts")
    p.add_argument("--pixel-size", type=int, default=32)
    args = p.parse_args()
    r = evaluate_run(args.export, args.out, args.data_dir, pixel_size=args.pixel_size)
    c = r["classification"]
    print(f"[{r['name']}] test top1={c['top1']:.3f} top5={c['top5']:.3f} macroF1={c['macro_f1']:.3f} ECE={c['ece']:.3f}")
    for gal, d in r["retrieval"]["embedding"].items():
        print(f"  retrieval galerie={gal:8s} cosine {d['cosine']}  euclid {d['euclid']}")
    if "pixels" in r["retrieval"]:
        print(f"  pixels (galerie=test) cosine {r['retrieval']['pixels']['test']['cosine']}")
    print(f"  majorité: {r['baselines']['majority']}")
    if r["analysis"]["species"]:
        s = r["analysis"]["species"]
        print(f"  erreurs: {s['same_species_errors']}/{s['n_errors']} restent dans la même espèce")
    print("  paires les plus confondues:")
    for m in c["most_confused"][:5]:
        print(f"    {m['true']} -> {m['pred']}: {m['count']} ({m['rate']:.0%})")


if __name__ == "__main__":
    main()