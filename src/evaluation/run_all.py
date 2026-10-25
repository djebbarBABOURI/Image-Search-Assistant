from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from src.export_format import load_export
from .aggregate import aggregate, to_markdown
from .evaluate import evaluate_run
from .qualitative import plot_errors, plot_retrieval, random_errors

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exports-dir", default="data")
    p.add_argument("--data-dir", default="data", help="racine des images (baseline pixels + figures)")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--pixel-size", type=int, default=32)
    p.add_argument("--n-errors", type=int, default=12)
    p.add_argument("--no-images", action="store_true", help="sans baseline pixels ni figures (images absentes)")
    args = p.parse_args()

    exports = sorted(Path(args.exports_dir).glob("export_*_seed*.npz"))
    if not exports:
        raise SystemExit(f"Aucun fichier export_<config>_seed<k>.npz dans {args.exports_dir}")
    data_dir = None if args.no_images else args.data_dir

    eval_files = []
    for path in exports:
        name = re.sub(r"^export_", "", path.stem)
        out = Path(args.runs_dir) / name
        r = evaluate_run(path, out, data_dir, pixel_size=args.pixel_size)
        c = r["classification"]
        print(f"[{name}] top1={c['top1']:.3f} top5={c['top5']:.3f} macroF1={c['macro_f1']:.3f}")
        if data_dir is not None:
            ex = load_export(path)
            seed = int(re.search(r"seed(\d+)$", name).group(1))
            rows = random_errors(ex, args.n_errors, seed)
            (out / "errors.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
            plot_errors(rows, data_dir, out / "errors.png")
            plot_retrieval(ex, data_dir, out / "retrieval_examples.png", seed=seed)
        eval_files.append(out / "eval.json")

    res = aggregate(eval_files)
    runs = Path(args.runs_dir)
    (runs / "summary.md").write_text(to_markdown(res) + "\n", encoding="utf-8")
    allkeys = sorted({k for g in res.values() for k in g})
    (runs / "summary_all.md").write_text(to_markdown(res, allkeys) + "\n", encoding="utf-8")
    (runs / "summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n" + to_markdown(res))
    print(f"\nTableaux écrits dans {runs}/summary.md et summary_all.md")

if __name__ == "__main__":
    main()
