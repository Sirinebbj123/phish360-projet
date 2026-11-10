"""Tâche 8 - modèle deep sur les captures d'écran : ResNet18 pré-entraîné (ImageNet), affiné.

Entrée : data/features/img_224 (tâche 4) = capture en 224 x 224, JPEG. Les images sont décodées
à la volée et normalisées avec la moyenne / l'écart-type d'ImageNet (prévu dès la tâche 4).

Le ResNet18 (18 couches de convolution, 11 M de paramètres) a appris sur ImageNet à reconnaître
formes, logos, textes, mises en page. On remplace sa dernière couche par une tête
(Dropout -> 256 -> ReLU -> 1) et on réentraîne tout le réseau avec un pas plus petit pour
le corps (1e-4) que pour la tête (1e-3).

Augmentation de données (entraînement seulement) : petit recadrage aléatoire et légères variations
de luminosité / contraste. Pas de miroir horizontal : il retournerait le texte et les logos.

Pages sans image exploitable (img_ok = 0) : exclues de l'entraînement ; au test, elles reçoivent
une image noire (la fusion, tâche 9, sait que l'image manque grâce à la colonne img_ok).

Sorties :
  results/deep_results.csv              une ligne par protocole (modele = img_resnet18)
  results/preds_img_resnet18.parquet    probabilité "phishing" des pages de test
  models/img_resnet18_<proto>.pt        poids (≈ 45 Mo)
  data/emb/img_<proto>.parquet          256 nombres + probabilité + img_ok par page (fusion, tâche 9)

Usage :
    python src/img_resnet18.py                           # 3 protocoles (≈ 10-20 min sur RTX 3050)
    python src/img_resnet18.py --only rq1 --epochs 2
    python src/img_resnet18.py --freeze                  # corps figé, seule la tête apprend (CPU possible)
    python src/img_resnet18.py --workers 0               # si Windows bloque sur le chargement des images
    python src/img_resnet18.py --no-amp                  # si "NaN" : calcul en fp32 (automatique sur GTX 16xx)
"""
import argparse
import io
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deep_common as dc  # noqa: E402

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   # ImageNet
EMB_DIM = 256
AMP = True          # précision mixte fp16 ; désactivée sur les GTX 16xx (NaN connus) ou avec --no-amp


def use_amp(device):
    return AMP and device == "cuda"


def transforms(train):
    from torchvision import transforms as T
    norm = [T.ToTensor(), T.Normalize(MEAN, STD)]
    if train:
        return T.Compose([T.RandomResizedCrop(224, scale=(0.80, 1.0), ratio=(0.9, 1.1)),
                          T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1), *norm])
    return T.Compose(norm)


class Screens(Dataset):
    """Décode les JPEG 224 x 224 à la demande (garder 10 000 images décodées en mémoire = 1,5 Go)."""

    def __init__(self, blobs, y, idx, train):
        self.blobs, self.y, self.idx, self.tf = blobs, y, idx, transforms(train)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = self.idx[k]
        b = self.blobs[i]
        im = Image.open(io.BytesIO(b)).convert("RGB") if b is not None else Image.new("RGB", (224, 224))
        if im.size != (224, 224):
            im = im.resize((224, 224))
        return self.tf(im), np.float32(self.y[i]), i


class ImgClassifier(nn.Module):
    def __init__(self, pretrained=True, dropout=0.3):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone.fc = nn.Identity()                        # sortie : vecteur de 512
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, EMB_DIM), nn.ReLU())
        self.head = nn.Linear(EMB_DIM, 1)

    def embed(self, x):
        return self.proj(self.backbone(x))

    def forward(self, x):
        return self.head(self.embed(x)).squeeze(1)


def loader(blobs, y, idx, train, args, device):
    return DataLoader(Screens(blobs, y, idx, train), batch_size=args.batch, shuffle=train,
                      num_workers=args.workers, pin_memory=device == "cuda",
                      persistent_workers=args.workers > 0)


@torch.no_grad()
def predict(model, dl, device):
    model.eval()
    ps, es, ii = [], [], []
    for x, _, i in dl:
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp(device)):
            e = model.embed(x.to(device, non_blocking=True))
            p = torch.sigmoid(model.head(e).squeeze(1))
        ps.append(p.float().cpu().numpy())
        es.append(e.float().cpu().numpy())
        ii.append(i.numpy())
    return np.concatenate(ps), np.concatenate(es), np.concatenate(ii)


def train(model, blobs, y, tr, va, args, device):
    if args.freeze:
        for prm in model.backbone.parameters():
            prm.requires_grad = False
    groups = [{"params": list(model.proj.parameters()) + list(model.head.parameters()), "lr": args.lr * 10}]
    if not args.freeze:
        groups.append({"params": model.backbone.parameters(), "lr": args.lr})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp(device))
    loss_fn = nn.BCEWithLogitsLoss()
    stop = dc.EarlyStopper(args.patience)
    dl_tr = loader(blobs, y, tr, True, args, device)
    dl_va = loader(blobs, y, va, False, args, device)
    for ep in range(1, args.epochs + 1):
        model.train()
        if args.freeze:
            model.backbone.eval()                               # BatchNorm figé aussi
        t0, total, n = time.perf_counter(), 0.0, 0
        for b, (x, yb, _) in enumerate(dl_tr):
            x, yb = x.to(device, non_blocking=True), yb.to(device)
            opt.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp(device)):
                loss = loss_fn(model(x), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total, n = total + loss.item() * len(x), n + len(x)
            if b % 50 == 0:
                print(f"    lot {b:4d}/{len(dl_tr)}  perte {total / n:.4f}", flush=True)
        sched.step()
        p_val, _, i_val = predict(model, dl_va, device)
        if not np.isfinite(p_val).all() or not np.isfinite(total):
            raise SystemExit("Le réseau produit des NaN (souvent la précision fp16 sur certaines cartes). "
                             "Relancez avec --no-amp, et si besoin --lr 5e-5.")
        score = average_precision_score(y[i_val], p_val)
        print(f"  époque {ep:2d} : perte {total / n:.4f}  PR-AUC validation {score:.4f}  ({time.perf_counter() - t0:.0f} s)")
        if stop.step(score, model, ep):
            print("  arrêt anticipé")
            break
        if args.pause:
            time.sleep(args.pause)
    model.load_state_dict(stop.state)
    print(f"  meilleure époque : {stop.epoch} (PR-AUC validation {stop.best:.4f})")
    return model


def load_images(df):
    img = pd.read_parquet(f"{dc.FEAT}/img_224", columns=["sample_id", "img", "img_ok"]).drop_duplicates("sample_id")
    d = df[["sample_id"]].merge(img, on="sample_id", how="left")
    ok = d.img_ok.fillna(0).astype(int).to_numpy()
    blobs = [b if (k == 1 and b is not None) else None for b, k in zip(d.img, ok)]
    return blobs, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=dc.PROTOCOLS)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4, help="pas du corps du réseau (la tête prend 10 x plus)")
    ap.add_argument("--workers", type=int, default=2, help="processus de décodage des images (0 = aucun)")
    ap.add_argument("--freeze", action="store_true", help="corps ResNet figé, seule la tête apprend")
    ap.add_argument("--no-pretrained", action="store_true", help="poids aléatoires (test hors ligne uniquement)")
    ap.add_argument("--pause", type=int, default=10)
    ap.add_argument("--no-amp", action="store_true", help="désactiver la précision mixte fp16 (calcul en fp32)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    global AMP
    device = dc.device_of(args.cpu)
    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        if args.no_amp or "GTX 16" in name:
            AMP = False
            print(f"Précision mixte désactivée ({'option --no-amp' if args.no_amp else name + ' : fp16 instable'}) : calcul en fp32")
    dc.setup_dirs()
    name = "img_resnet18_gele" if args.freeze else "img_resnet18"
    df = dc.load_index()
    blobs, img_ok = load_images(df)
    y = df.y.to_numpy()
    print(f"{len(df)} pages, {img_ok.sum()} images exploitables ({(1 - img_ok.mean()):.1%} manquantes)")

    rows, preds = [], []
    for proto in dc.protocols_arg(args.only):
        part = dc.protocol_parts(df, proto)
        has = img_ok == 1
        tr = np.flatnonzero((part == "train") & has)          # on n'apprend pas sur des images noires
        va = np.flatnonzero((part == "val") & has)
        te = np.flatnonzero(part == "test")
        used = np.flatnonzero(part != "hors")
        print(f"\n=== {proto} : {len(tr)} images d'entraînement, {len(va)} de validation, {len(te)} de test ===")
        dc.seed_all(42)
        model = ImgClassifier(pretrained=not args.no_pretrained).to(device)
        with dc.Timer() as t:
            model = train(model, blobs, y, tr, va, args, device)
        with dc.Timer() as ti:
            p_used, e_used, i_used = predict(model, loader(blobs, y, used, False, args, device), device)
        order = np.argsort(i_used)                               # remettre dans l'ordre de `used`
        p_used, e_used = p_used[order], e_used[order]
        p = np.full(len(df), np.nan)
        p[used] = p_used
        thr = dc.best_threshold(y[va], p[va])
        m = dc.metrics(y[te], p[te], thr)
        rows.append({"protocole": proto, "modele": name, **m, "train_s": round(t.s, 1), "appareil": device,
                     "inference_pages_s": round(len(used) / ti.s, 1)})
        print(name, dc.fmt(m), f"({t.s:.0f} s)")
        dc.log_time(f"deep_{proto}_{name}", len(tr), t.s, device)
        preds.append(pd.DataFrame({"sample_id": df.sample_id.to_numpy()[te], "protocole": proto, "modele": name,
                                   "y": y[te], "p_phishing": p[te]}))
        torch.save(model.state_dict(), f"models/{name}_{proto}.pt")
        path = dc.save_embeddings(df.iloc[used], part[used], e_used, p_used, "img", proto, "i")
        emb = pd.read_parquet(path)
        emb.insert(4, "img_ok", img_ok[used])
        emb.to_parquet(path, index=False)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    dc.save_results(rows)
    dc.save_preds(preds, name)
    print("\nRésultats image :")
    print(pd.DataFrame(rows)[["protocole", "modele", "f1", "precision", "recall", "roc_auc", "pr_auc"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
