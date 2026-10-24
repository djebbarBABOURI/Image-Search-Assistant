from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
import numpy as np

KEY_METRICS = [
    "classification.top1", "classification.top5", "classification.macro_f1", "classification.ece",
    "retrieval.embedding.test.cosine.p@1", "retrieval.embedding.test.cosine.p@5",
    "retrieval.embedding.test.cosine.p@10", "retrieval.embedding.test.euclid.p@5",
    "retrieval.embedding.trainval.cosine.p@5", "retrieval.pixels.test.cosine.p@5",
    "baselines.majority.top1", "baselines.random_retrieval.p@5",
    "analysis.species.species_accuracy", "analysis.species.same_species_error_share",
    "analysis.confidence.t80.coverage", "analysis.confidence.t80.accuracy",
]

def is_percentage_metric(key: str) -> bool:
    if key.endswith(".ece") or key in {"n_test", "embedding_dim", "classification.n"}:
        return False
    if key.endswith(".n") or key.endswith(".support") or key.endswith("_errors"):
        return False
    return (
        key.endswith((".top1", ".top5", ".macro_f1", ".macro_precision", ".macro_recall",
                      ".mean_confidence", ".accuracy", ".coverage", ".species_accuracy",
                      ".same_species_error_share"))
        or ".p@" in key
    )

def flatten(d, prefix="") -> dict[str, float]:
    out = {}
    for k, v in d.items():
        if k in ("per_class", "most_confused", "confusion_matrix"):
            continue
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
    return out

def group_name(path: Path) -> str:
    return re.sub(r"_seed\d+$", "", path.parent.name)

def aggregate(paths) -> dict[str, dict[str, tuple[float, float, int]]]:
    groups = defaultdict(list)
    for p in map(Path, paths):
        groups[group_name(p)].append(flatten(json.loads(p.read_text(encoding="utf-8"))))
    res = {}
    for group, runs in groups.items():
        res[group] = {}
        for key in sorted(set().union(*runs)):
            vals = np.array([r[key] for r in runs if key in r], dtype=float)
            std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            res[group][key] = (float(vals.mean()), std, len(vals))
    return res

def format_metric(key: str, mean: float, std: float) -> str:
    if is_percentage_metric(key):
        return f"{mean * 100:.1f} ± {std * 100:.1f} %"
    if key in {"n_test", "embedding_dim", "classification.n"} or key.endswith(".n") or key.endswith("_errors") or key.endswith(".support"):
        return f"{mean:.1f} ± {std:.1f}"
    return f"{mean:.3f} ± {std:.3f}"

def to_markdown(res, keys=None) -> str:
    groups = list(res)
    if not groups:
        return "| Métrique |\n|---|"
    keys = list(keys) if keys is not None else KEY_METRICS
    keys = [k for k in keys if any(k in res[g] for g in groups)]
    n_runs = {g: max((v[2] for v in res[g].values()), default=0) for g in groups}
    lines = [
        "| Métrique | " + " | ".join(f"{g} (n={n_runs[g]})" for g in groups) + " |",
        "|---|" + "---|" * len(groups),
    ]
    for key in keys:
        cells = []
        for group in groups:
            if key not in res[group]:
                cells.append("–")
            else:
                mean, std, _ = res[group][key]
                cells.append(format_metric(key, mean, std))
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_files", nargs="+")
    parser.add_argument("--out", default=None, help="fichier .md (et .json à côté)")
    parser.add_argument("--all", action="store_true", help="toutes les métriques, pas seulement les clés principales")
    args = parser.parse_args()
    res = aggregate(args.eval_files)
    keys = sorted({key for group in res.values() for key in group}) if args.all else None
    md = to_markdown(res, keys)
    print(md)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md + "\n", encoding="utf-8")
        out.with_suffix(".json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    main()
