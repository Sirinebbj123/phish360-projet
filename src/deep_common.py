"""Outils communs aux scripts deep du Sprint 2 (tâches 6 à 12).

Tout ce qui doit être IDENTIQUE d'un modèle à l'autre est ici, pour que les comparaisons soient justes :
  - les protocoles (quelles pages servent à entraîner / tester) ;
  - la validation : les mêmes 10 % de pages d'entraînement sont mises de côté pour TOUS les modèles
    (choix déterministe à partir du sample_id). Une page de validation n'est donc jamais vue
    par aucun encodeur : c'est sur elle que la fusion "stacking" (tâche 9) apprend ses poids ;
  - les métriques (F1, précision, rappel, ROC-AUC, PR-AUC, seuil choisi sur la validation) ;
  - le format des fichiers produits (results/deep_results.csv, results/preds_<modele>.parquet,
    data/emb/<modalite>_<protocole>.parquet).

Protocoles :
  rq1      trainval -> test (découpage officiel de Phish360)
  rq2      période "ancienne" -> période "recente" (tâche 3, année de copyright des kits)
  rq2ctrl  contrôle de RQ2 : les MÊMES pages que RQ2, mais réparties au hasard avec les mêmes tailles
           (entraînement = taille de "ancienne"). Si un modèle est bien meilleur en rq2ctrl qu'en rq2,
           sa baisse en RQ2 vient du temps (nouvelles campagnes) et pas du manque de données.
"""
import csv
import hashlib
import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)

PROTOCOLS = ["rq1", "rq2", "rq2ctrl"]
FEAT = "data/features"
EMB = "data/emb"
VAL_SHARE = 10            # % des pages d'entraînement gardées pour la validation


# ---------- données et protocoles ------------------------------------------------

def load_index():
    """Une ligne par page gardée : sample_id, split, label, y (1 = phishing), rq2_period. Triée par sample_id."""
    url = pd.read_parquet(f"{FEAT}/url_feat", columns=["sample_id", "split", "label"])
    rq2 = pd.read_parquet("data/rq2_split.parquet", columns=["sample_id", "rq2_period"])
    df = url.merge(rq2, on="sample_id", how="left").drop_duplicates("sample_id")
    df["y"] = (df.label == "phishing").astype(int)
    return df.sort_values("sample_id").reset_index(drop=True)


def is_val(sample_ids):
    """True pour ~10 % des pages, toujours les mêmes (hachage du sample_id, indépendant de l'ordre des lignes)."""
    return np.array([int(hashlib.md5(s.encode()).hexdigest(), 16) % 100 < VAL_SHARE for s in sample_ids])


def protocol_parts(df, proto, seed=42):
    """Renvoie un tableau 'part' aligné sur df : train / val / test / hors (page non utilisée)."""
    if proto == "rq1":
        tr, te = (df.split == "trainval").to_numpy(), (df.split == "test").to_numpy()
    elif proto == "rq2":
        tr, te = (df.rq2_period == "ancienne").to_numpy(), (df.rq2_period == "recente").to_numpy()
    elif proto == "rq2ctrl":
        pool = df.rq2_period.notna().to_numpy()
        n_train = int((df.rq2_period == "ancienne").sum())
        rng = np.random.default_rng(seed)
        tr, te = np.zeros(len(df), bool), np.zeros(len(df), bool)
        # tirage stratifié par classe : même part de phishing dans l'entraînement et le test
        for cls in (0, 1):
            idx = np.flatnonzero(pool & (df.y.to_numpy() == cls))
            idx = rng.permutation(idx)
            k = round(len(idx) * n_train / pool.sum())
            tr[idx[:k]], te[idx[k:]] = True, True
    else:
        raise ValueError(proto)
    part = np.full(len(df), "hors", dtype=object)
    part[tr] = "train"
    part[tr & is_val(df.sample_id)] = "val"
    part[te] = "test"
    return part


def protocols_arg(only):
    return [only] if only else PROTOCOLS


# ---------- métriques ------------------------------------------------------------

def best_threshold(y, p):
    """Seuil qui maximise le F1 (calculé sur la validation, jamais sur le test)."""
    if len(set(y)) < 2:
        return 0.5
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    return float(thr[np.argmax(f1[:-1])]) if len(thr) else 0.5


def metrics(y, p, thr=0.5):
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= 0.5).astype(int)
    two = len(set(y)) > 1
    return {"n_test": len(y), "f1": f1_score(y, pred, zero_division=0),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, p) if two else float("nan"),
            "pr_auc": average_precision_score(y, p) if two else float("nan"),
            "seuil_val": thr, "f1_seuil_val": f1_score(y, (p >= thr).astype(int), zero_division=0)}


def fmt(m):
    return (f"F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}  ROC-AUC={m['roc_auc']:.3f}  "
            f"PR-AUC={m['pr_auc']:.3f}  | seuil val {m['seuil_val']:.2f} -> F1={m['f1_seuil_val']:.3f}")


# ---------- sorties --------------------------------------------------------------

def setup_dirs():
    for d in ["results", "results/figures", "models", EMB]:
        os.makedirs(d, exist_ok=True)


def save_results(rows, path="results/deep_results.csv"):
    """Ajoute les lignes (protocole, modele) ; une ligne existante avec le même couple est remplacée."""
    new = pd.DataFrame(rows)
    if os.path.exists(path):
        old = pd.read_csv(path)
        key_new = set(zip(new.protocole, new.modele))
        old = old[[(p, m) not in key_new for p, m in zip(old.protocole, old.modele)]]
        new = pd.concat([old, new], ignore_index=True)
    new.round(4).to_csv(path, index=False)


def save_preds(frames, modele):
    """results/preds_<modele>.parquet : sample_id, protocole, modele, y, p_phishing (pages de test).
    Les protocoles non relancés cette fois sont conservés."""
    path = f"results/preds_{modele}.parquet"
    new = pd.concat(frames, ignore_index=True)
    if os.path.exists(path):
        old = pd.read_parquet(path)
        new = pd.concat([old[~old.protocole.isin(new.protocole)], new], ignore_index=True)
    new.to_parquet(path, index=False)


def save_embeddings(df, part, emb, p, modalite, proto, prefix):
    """data/emb/<modalite>_<proto>.parquet : sample_id, part, y, p_phishing, <prefix>0..<prefix>(d-1)."""
    out = pd.DataFrame(emb.astype(np.float32), columns=[f"{prefix}{i}" for i in range(emb.shape[1])])
    out.insert(0, "p_phishing", p.astype(np.float32))
    out.insert(0, "y", df.y.to_numpy())
    out.insert(0, "part", part)
    out.insert(0, "sample_id", df.sample_id.to_numpy())
    path = f"{EMB}/{modalite}_{proto}.parquet"
    out.to_parquet(path, index=False)
    return path


def log_time(step, pages, secs, device):
    """Même format que results/timings.csv des tâches 2 à 5 (colonne 'coeurs' = appareil pour le deep)."""
    path = "results/timings.csv"
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "etape", "pages", "coeurs", "secondes", "pages_par_s"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), step, pages, device,
                    round(secs, 2), round(pages / secs, 1) if secs else 0])


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_of(force_cpu=False):
    import torch
    dev = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
    name = torch.cuda.get_device_name(0) if dev == "cuda" else "CPU"
    print(f"Appareil : {dev} ({name})")
    return dev


class EarlyStopper:
    """Garde le meilleur état du modèle selon la PR-AUC de validation ; arrête après `patience` époques sans progrès."""

    def __init__(self, patience):
        self.patience, self.best, self.state, self.wait, self.epoch = patience, -1.0, None, 0, 0

    def step(self, score, model, epoch):
        if score > self.best + 1e-4:
            self.best, self.wait, self.epoch = score, 0, epoch
            self.state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.wait += 1
        return self.wait >= self.patience


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.s = time.perf_counter() - self.t0
