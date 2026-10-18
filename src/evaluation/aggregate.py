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

def flatten(d, prefix="") -> dict[str, float]:
    out = {}
    for k, v in d.items():
        if k in ("per_class", "most_confused"):
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
        groups[group_name(p)].append(flatten(json.loads(p.read_text())))
    res = {}
    for g, runs in groups.items():
        res[g] = {}
        for key in sorted(set().union(*runs)):
            vals = np.array([r[key] for r in runs if key in r])
            std = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
            res[g][key] = (float(vals.mean()), std, len(vals))
    return res

def to_markdown(res, keys=None, as_percent=True) -> str:
    groups = list(res)
    keys = keys or KEY_METRICS
    keys = [k for k in keys if any(k in res[g] for g in groups)]
    lines = ["| Métrique | " + " | ".join(f"{g} (n={max(res[g][k][2] for k in res[g])})" for g in groups) + " |",
             "|---|" + "---|" * len(groups)]
    for k in keys:
        cells = []
        for g in groups:
            if k not in res[g]:
                cells.append("–")
                continue
            m, s, _ = res[g][k]
            f = 100 if as_percent and "ece" not in k else 1
            fmt = "{:.1f} ± {:.1f}" if f == 100 else "{:.3f} ± {:.3f}"
            cells.append(fmt.format(m * f, s * f) + (" %" if f == 100 else ""))
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    return "\n".join(lines)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("eval_files", nargs="+")
    p.add_argument("--out", default=None, help="fichier .md (et .json à côté)")
    p.add_argument("--all", action="store_true", help="toutes les métriques, pas seulement les clés")
    args = p.parse_args()
    res = aggregate(args.eval_files)
    keys = sorted({k for g in res.values() for k in g}) if args.all else None
    md = to_markdown(res, keys)
    print(md)
    if args.out:
        Path(args.out).write_text(md + "\n")
        Path(args.out).with_suffix(".json").write_text(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()