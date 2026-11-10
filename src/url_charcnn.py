"""Tâche 6 - modèle deep sur l'URL : CharCNN (réseau de convolution sur les caractères).

L'URL est lue caractère par caractère : chaque caractère devient un petit vecteur appris
(embedding), puis des filtres de convolution de largeur 3, 4 et 5 repèrent des motifs
("pay", "-log", ".xyz", "//www."...). Un max-pooling garde le motif le plus fort de chaque filtre.

Version Sprint 2 : utilise src/deep_common.py, comme les tâches 7 à 9, pour que TOUS les modèles
deep aient exactement la même validation (les mêmes 10 % de pages) et les mêmes protocoles :
  rq1      trainval -> test
  rq2      période "ancienne" -> "recente"
  rq2ctrl  mêmes pages que rq2, réparties au hasard (contrôle de taille, tâche 10)
=> à relancer une fois (moins de 2 minutes sur GPU) avant la fusion (tâche 9).

Sorties :
  results/deep_results.csv           une ligne par protocole (modele = url_charcnn)
  results/preds_url_charcnn.parquet  probabilité "phishing" de chaque page de test
  models/url_charcnn_<proto>.pt      poids du modèle
  data/emb/url_<proto>.parquet       128 nombres + probabilité par page (entrée de la fusion, tâche 9)

Usage :
    python src/url_charcnn.py                       # GPU si disponible, 3 protocoles
    python src/url_charcnn.py --only rq1 --epochs 5
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deep_common as dc  # noqa: E402

MAX_LEN = 200                                           # 99 % des URL font moins de 200 caractères
VOCAB = {c: i + 2 for i, c in enumerate(map(chr, range(32, 127)))}   # 0 = remplissage, 1 = caractère inconnu


def encode(url):
    ids = [VOCAB.get(c, 1) for c in (url or "").lower()[:MAX_LEN]]
    return ids + [0] * (MAX_LEN - len(ids))


class CharCNN(nn.Module):
    def __init__(self, emb=32, filters=128, out=128):
        super().__init__()
        self.emb = nn.Embedding(len(VOCAB) + 2, emb, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb, filters, k) for k in (3, 4, 5)])
        self.proj = nn.Sequential(nn.Dropout(0.5), nn.Linear(3 * filters, out), nn.ReLU())
        self.head = nn.Linear(out, 1)

    def embed(self, x):                                 # (batch, 200) -> (batch, 128)
        h = self.emb(x).transpose(1, 2)                 # (batch, 32, 200) : Conv1d veut les canaux en 2e
        h = torch.cat([torch.relu(c(h)).amax(dim=2) for c in self.convs], dim=1)
        return self.proj(h)

    def forward(self, x):
        return self.head(self.embed(x)).squeeze(1)      # un score (logit) par URL


def batches(X, y, size, shuffle, rng=None):
    idx = rng.permutation(len(X)) if shuffle else np.arange(len(X))
    for i in range(0, len(X), size):
        j = idx[i:i + size]
        yield X[j], (y[j] if y is not None else None)


def predict(model, X, device, size=512, emb=False):
    """Probabilités "phishing" (et, si emb=True, les vecteurs de 128 nombres)."""
    model.eval()
    ps, es = [], []
    with torch.no_grad():
        for xb, _ in batches(X, None, size, False):
            e = model.embed(torch.from_numpy(xb).to(device))
            ps.append(torch.sigmoid(model.head(e).squeeze(1)).cpu().numpy())
            es.append(e.cpu().numpy())
    p = np.concatenate(ps) if ps else np.array([])
    return (p, np.concatenate(es)) if emb else p


def train(Xtr, ytr, Xva, yva, args, device, seed=42):
    """Entraîne sur la partie 'train' et garde le meilleur modèle selon la PR-AUC de la partie 'val'."""
    dc.seed_all(seed)
    rng = np.random.default_rng(seed)
    model = CharCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    stop = dc.EarlyStopper(args.patience)
    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        with dc.Timer() as t:
            for xb, yb in batches(Xtr, ytr, args.batch, True, rng):
                xb, yb = torch.from_numpy(xb).to(device), torch.from_numpy(yb).float().to(device)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
                total += loss.item() * len(xb)
        score = average_precision_score(yva, predict(model, Xva, device))
        print(f"  époque {ep:2d} : perte {total / len(Xtr):.4f}  PR-AUC validation {score:.4f}  ({t.s:.0f} s)")
        if stop.step(score, model, ep):
            print("  arrêt anticipé")
            break
        if args.pause:
            import time
            time.sleep(args.pause)                      # laisser souffler le GPU
    model.load_state_dict(stop.state)
    print(f"  meilleure époque : {stop.epoch} (PR-AUC validation {stop.best:.4f})")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=dc.PROTOCOLS)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=4, help="époques sans progrès avant d'arrêter")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pause", type=int, default=3, help="secondes de pause entre deux époques")
    ap.add_argument("--cpu", action="store_true", help="forcer le CPU")
    args = ap.parse_args()

    device = dc.device_of(args.cpu)
    dc.setup_dirs()
    df = dc.load_index()
    url = pd.read_parquet(f"{dc.FEAT}/url_feat", columns=["sample_id", "url"]).drop_duplicates("sample_id")
    df = df.merge(url, on="sample_id", how="left")
    X = np.array([encode(u) for u in df.url], dtype=np.int64)
    y = df.y.to_numpy()

    rows, preds = [], []
    for proto in dc.protocols_arg(args.only):
        part = dc.protocol_parts(df, proto)
        tr, va, te = part == "train", part == "val", part == "test"
        print(f"\n=== {proto} : {tr.sum()} URL d'entraînement, {va.sum()} de validation, {te.sum()} de test ===")
        with dc.Timer() as t:
            model = train(X[tr], y[tr], X[va], y[va], args, device)
        used = part != "hors"
        p_all, e_all = predict(model, X[used], device, emb=True)
        p = np.full(len(df), np.nan)
        p[used] = p_all
        thr = dc.best_threshold(y[va], p[va])
        m = dc.metrics(y[te], p[te], thr)
        rows.append({"protocole": proto, "modele": "url_charcnn", **m, "train_s": round(t.s, 1), "appareil": device})
        print("url_charcnn", dc.fmt(m), f"({t.s:.0f} s)")
        dc.log_time(f"deep_{proto}_url_charcnn", int(tr.sum()), t.s, device)
        preds.append(pd.DataFrame({"sample_id": df.sample_id[te].to_numpy(), "protocole": proto,
                                   "modele": "url_charcnn", "y": y[te], "p_phishing": p[te]}))
        torch.save(model.state_dict(), f"models/url_charcnn_{proto}.pt")
        dc.save_embeddings(df[used], part[used], e_all, p_all, "url", proto, "u")

    dc.save_results(rows)
    dc.save_preds(preds, "url_charcnn")

    print("\nComparaison avec la baseline URL TF-IDF (tâche 5) :")
    res = pd.DataFrame(rows)
    if os.path.exists("results/baselines.csv"):
        base = pd.read_csv("results/baselines.csv").query("modele == 'url_tfidf'")
        res = pd.concat([base, res])
    print(res[["protocole", "modele", "f1", "precision", "recall", "pr_auc"]]
          .sort_values(["protocole", "modele"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
