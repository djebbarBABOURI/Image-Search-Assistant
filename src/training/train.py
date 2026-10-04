"""
Entraînement de ResNet-18 sur Oxford-IIIT Pets (37 races de chats et chiens).

Le même script sert pour les deux modes, pour que la comparaison soit juste :
    python train.py --mode finetune --seed 0     # poids pré-entraînés ImageNet
    python train.py --mode scratch  --seed 0     # poids aléatoires

Résultats sauvegardés dans runs/<mode>_seed<seed>/ :
    model.pt      -> meilleur modèle (selon la validation)
    results.json  -> courbes par époque + accuracy finale sur le test
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

# Moyenne / écart-type d'ImageNet (normalisation standard pour ResNet)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SPLIT_SEED = 42  # découpage train/val FIXE, identique pour tout le groupe


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["finetune", "scratch"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None,
                   help="par défaut : 1e-4 en finetune, 1e-3 en scratch")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="runs")
    return p.parse_args()


# ---------------------------------------------------------------- données
def get_loaders(data_dir, batch_size):
    # Augmentation légère pour l'entraînement
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    # Pas d'augmentation pour validation / test
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # Téléchargement automatique la première fois (~800 Mo)
    trainval_aug = datasets.OxfordIIITPet(data_dir, split="trainval",
                                          transform=train_tf, download=True)
    trainval_eval = datasets.OxfordIIITPet(data_dir, split="trainval",
                                           transform=eval_tf, download=True)
    test_set = datasets.OxfordIIITPet(data_dir, split="test",
                                      transform=eval_tf, download=True)

    # 85 % train / 15 % validation, toujours le même découpage
    n = len(trainval_aug)
    n_val = int(0.15 * n)
    g = torch.Generator().manual_seed(SPLIT_SEED)
    idx = torch.randperm(n, generator=g).tolist()
    train_set = Subset(trainval_aug, idx[n_val:])
    val_set = Subset(trainval_eval, idx[:n_val])

    kw = dict(batch_size=batch_size, num_workers=2)
    return (DataLoader(train_set, shuffle=True, **kw),
            DataLoader(val_set, shuffle=False, **kw),
            DataLoader(test_set, shuffle=False, **kw),
            trainval_aug.classes)


# ---------------------------------------------------------------- modèle
def build_model(mode, num_classes):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if mode == "finetune" else None
    model = models.resnet18(weights=weights)
    # On remplace la dernière couche (1000 classes ImageNet -> 37 races)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ---------------------------------------------------------------- boucles
def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for images, labels in loader:
        optimizer.zero_grad()
        outputs = model(images)            # 1. prédiction
        loss = loss_fn(outputs, labels)    # 2. erreur (loss)
        loss.backward()                    # 3. calcul des gradients
        optimizer.step()                   # 4. mise à jour des poids
        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        n += labels.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss, top1, top5, n = 0.0, 0, 0, 0
    for images, labels in loader:
        outputs = model(images)
        total_loss += loss_fn(outputs, labels).item() * labels.size(0)
        top1 += (outputs.argmax(1) == labels).sum().item()
        top5 += (outputs.topk(5, dim=1).indices == labels.unsqueeze(1)).any(1).sum().item()
        n += labels.size(0)
    return total_loss / n, top1 / n, top5 / n


# ---------------------------------------------------------------- main
def main():
    args = get_args()
    torch.manual_seed(args.seed)
    lr = args.lr or (1e-4 if args.mode == "finetune" else 1e-3)

    out = Path(args.out_dir) / f"{args.mode}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, classes = get_loaders(args.data_dir, args.batch_size)
    model = build_model(args.mode, len(classes))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history, best_val = [], -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, optimizer)
        va_loss, va_acc, va_top5 = evaluate(model, val_loader, loss_fn)
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc, val_top5=va_top5))
        print(f"[{args.mode} seed={args.seed}] époque {epoch}/{args.epochs} "
              f"train_acc={tr_acc:.3f} val_acc={va_acc:.3f} ({time.time() - t0:.0f}s)")
        if va_acc > best_val:  # on garde le meilleur modèle sur la validation
            best_val = va_acc
            torch.save(model.state_dict(), out / "model.pt")

    # Évaluation finale sur le test avec le meilleur modèle
    model.load_state_dict(torch.load(out / "model.pt"))
    te_loss, te_top1, te_top5 = evaluate(model, test_loader, loss_fn)
    print(f"TEST top1={te_top1:.3f} top5={te_top5:.3f}")

    results = dict(mode=args.mode, seed=args.seed, epochs=args.epochs, lr=lr,
                   batch_size=args.batch_size, best_val_acc=best_val,
                   test_top1=te_top1, test_top5=te_top5,
                   history=history, classes=classes)
    (out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
