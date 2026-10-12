"""
Entraînement de ResNet-18 sur Oxford-IIIT Pets (37 races de chats et chiens).

Le même script sert pour les deux modes, pour que la comparaison soit juste :
    python -m src.training.train --mode finetune --seed 0 --epochs 50 --out-dir runs_50ep
    python -m src.training.train --mode scratch  --seed 0 --epochs 50 --out-dir runs_50ep

Réglages communs (identiques pour les deux modes, sauf le learning rate de départ) :
    - AdamW, weight decay 0.01
    - scheduler cosine : le learning rate diminue progressivement jusqu'à ~0
    - early stopping : arrêt si la validation ne s'améliore pas pendant `patience` époques

Options d'amélioration (désactivées par défaut, pour garder les anciens réglages reproductibles) :
    --augment trivial        TrivialAugmentWide au lieu de ColorJitter + rotation
    --label-smoothing 0.1    le modèle vise 90 % de certitude au lieu de 100 %
    --mixup                  MixUp ou CutMix (au hasard) sur chaque batch d'entraînement

Résultats sauvegardés dans <out-dir>/<mode>_seed<seed>/ :
    model.pt      -> meilleur modèle (selon la validation)
    results.json  -> réglages, courbes par époque, époque d'arrêt, accuracy sur le test
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.transforms import v2

# Moyenne / écart-type d'ImageNet (normalisation standard pour ResNet)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SPLIT_SEED = 42  # découpage train/val FIXE, identique pour tout le groupe
NUM_CLASSES = 37


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["finetune", "scratch"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None,
                   help="lr de départ ; par défaut : 1e-4 en finetune, 1e-3 en scratch")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=10,
                   help="early stopping : nb d'époques sans amélioration de la val (0 = désactivé)")
    p.add_argument("--augment", choices=["standard", "trivial"], default="standard",
                   help="standard = ColorJitter + rotation ; trivial = TrivialAugmentWide")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--mixup", action="store_true", help="active MixUp / CutMix")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="runs")
    return p.parse_args()


# ---------------------------------------------------------------- données
def get_loaders(data_dir, batch_size, augment):
    # Augmentation UNIQUEMENT pour l'entraînement
    if augment == "trivial":
        # une transformation aléatoire parmi ~15 (contraste, netteté, cisaillement...)
        extra = [transforms.TrivialAugmentWide()]
    else:
        # teinte (hue) très faible : la couleur du pelage aide à reconnaître la race
        extra = [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
                 transforms.RandomRotation(15)]
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        *extra,
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.15),  # après ToTensor : travaille sur des tenseurs
    ])
    # Pas d'augmentation pour validation / test (mais la même normalisation)
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
    return (DataLoader(train_set, shuffle=True, drop_last=True, **kw),
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
def train_one_epoch(model, loader, loss_fn, optimizer, mix=None):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for images, labels in loader:
        true_labels = labels
        if mix is not None:
            # MixUp / CutMix : mélange deux images et leurs étiquettes (étiquettes "douces")
            images, labels = mix(images, labels)
            true_labels = labels.argmax(1)  # classe dominante, pour l'accuracy indicative
        optimizer.zero_grad()
        outputs = model(images)            # 1. prédiction
        loss = loss_fn(outputs, labels)    # 2. erreur (loss)
        loss.backward()                    # 3. calcul des gradients
        optimizer.step()                   # 4. mise à jour des poids
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == true_labels).sum().item()
        n += images.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()  # loss "normale" en évaluation (sans label smoothing)
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

    train_loader, val_loader, test_loader, classes = get_loaders(
        args.data_dir, args.batch_size, args.augment)
    model = build_model(args.mode, len(classes))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    mix = (v2.RandomChoice([v2.MixUp(num_classes=NUM_CLASSES),
                            v2.CutMix(num_classes=NUM_CLASSES)])
           if args.mixup else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    # Cosine : le lr part de `lr` et descend progressivement jusqu'à ~0 à la dernière époque
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history, best_val, best_epoch = [], -1.0, 0
    stopped_epoch = args.epochs
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, mix)
        va_loss, va_acc, va_top5 = evaluate(model, val_loader)
        scheduler.step()
        history.append(dict(epoch=epoch, lr=current_lr, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc, val_top5=va_top5))
        print(f"[{args.mode} seed={args.seed}] époque {epoch}/{args.epochs} "
              f"lr={current_lr:.2e} train_acc={tr_acc:.3f} val_acc={va_acc:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)

        if va_acc > best_val:  # on garde le meilleur modèle sur la validation
            best_val, best_epoch = va_acc, epoch
            torch.save(model.state_dict(), out / "model.pt")
        elif args.patience and epoch - best_epoch >= args.patience:
            stopped_epoch = epoch
            print(f"Early stopping : pas d'amélioration depuis {args.patience} époques "
                  f"(meilleure val = {best_val:.3f} à l'époque {best_epoch})")
            break

    # Évaluation finale sur le test avec le meilleur modèle
    model.load_state_dict(torch.load(out / "model.pt"))
    te_loss, te_top1, te_top5 = evaluate(model, test_loader)
    print(f"TEST top1={te_top1:.3f} top5={te_top5:.3f} (meilleur modèle : époque {best_epoch})")

    results = dict(mode=args.mode, seed=args.seed, epochs=args.epochs, lr=lr,
                   weight_decay=args.weight_decay, scheduler="cosine",
                   patience=args.patience, batch_size=args.batch_size,
                   augment=args.augment, label_smoothing=args.label_smoothing,
                   mixup=args.mixup,
                   best_val_acc=best_val, best_epoch=best_epoch, stopped_epoch=stopped_epoch,
                   test_top1=te_top1, test_top5=te_top5,
                   history=history, classes=classes)
    (out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
    # Force la fin du programme (évite le blocage après TEST sur Mac) :
    # les résultats sont déjà enregistrés à ce stade.
    os._exit(0)