"""Export au format du contrat (src/export_format.py) à partir d'un run entraîné.

    python -m src.training.export --run runs/finetune_seed0 --out data/export.npz

Produit pour TOUTES les images (trainval + test) : probabilités softmax, embedding
(sortie du ResNet-18 avant la couche fc, 512 dimensions), vraie classe et split.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.export_format import Export, save_export
from src.training.train import MEAN, SPLIT_SEED, STD, build_model


@torch.no_grad()
def run_model(model, loader):
    """Retourne (embeddings, probs, labels) dans l'ordre du dataset."""
    model.eval()
    fc = model.fc
    model.fc = nn.Identity()  # le modèle sort maintenant l'embedding (N, 512)
    embs, probs, labels = [], [], []
    for images, y in loader:
        f = model(images)
        embs.append(f)
        probs.append(fc(f).softmax(1))
        labels.append(y)
    model.fc = fc
    return (torch.cat(embs).numpy(), torch.cat(probs).numpy(), torch.cat(labels).numpy())


def relative_paths(ds, data_dir):
    return [os.path.relpath(p, data_dir).replace(os.sep, "/") for p in ds._images]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="dossier d'un run, ex. runs/finetune_seed0")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="data/export.npz")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    run = Path(args.run)
    results = json.loads((run / "results.json").read_text())
    classes = results["classes"]

    model = build_model(results["mode"], len(classes))
    # model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    # Chercher d'abord le checkpoint dans le dossier du run, puis dans models/ si nécessaire.
    model_path = run / "model.pt"

    if not model_path.exists():
        model_path = Path("models/model.pt")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Checkpoint introuvable : {run / 'model.pt'} "
            "et models/model.pt"
        )

    print(f"Chargement du modèle : {model_path}")
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    eval_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    trainval = datasets.OxfordIIITPet(args.data_dir, split="trainval", transform=eval_tf, download=True)
    test = datasets.OxfordIIITPet(args.data_dir, split="test", transform=eval_tf, download=True)

    # Même découpage train/val que dans train.py (SPLIT_SEED fixe)
    n = len(trainval)
    n_val = int(0.15 * n)
    g = torch.Generator().manual_seed(SPLIT_SEED)
    val_idx = set(torch.randperm(n, generator=g).tolist()[:n_val])
    split_tv = np.array(["val" if i in val_idx else "train" for i in range(n)])

    parts = []
    for ds, split in ((trainval, split_tv), (test, np.array(["test"] * len(test)))):
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
        emb, probs, labels = run_model(model, loader)
        parts.append((relative_paths(ds, args.data_dir), emb, probs, labels, split))

    export = Export(
        task="single",
        image_paths=np.array(sum((x[0] for x in parts), [])),
        class_names=np.array(classes),
        embeddings=np.concatenate([x[1] for x in parts]),
        probs=np.concatenate([x[2] for x in parts]),
        labels=np.concatenate([x[3] for x in parts]),
        split=np.concatenate([x[4] for x in parts]),
    )
    save_export(args.out, export)

    test_mask = export.split == "test"
    acc = (export.probs[test_mask].argmax(1) == export.labels[test_mask]).mean()
    print(f"Export écrit dans {args.out} : {len(export.image_paths)} images, "
          f"embeddings {export.embeddings.shape[1]}d, accuracy test = {acc:.3f} "
          f"(doit correspondre à test_top1 = {results['test_top1']:.3f})")


if __name__ == "__main__":
    main()
