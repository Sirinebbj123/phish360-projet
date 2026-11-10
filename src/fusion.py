"""Tâche 9 - fusion des 4 modalités (RQ1, et RQ2 pour la robustesse dans le temps).

Entrées (une par modalité, produites par les tâches 6, 7, 8 et 4) :
  url   data/emb/url_<proto>.parquet    128 nombres (CharCNN)     + sa probabilité
  html  data/emb/html_<proto>.parquet   256 nombres (DistilBERT)  + sa probabilité
  img   data/emb/img_<proto>.parquet    256 nombres (ResNet18)    + sa probabilité + img_ok
  tech  data/features/tech_feat         35 features techniques (log + standardisation)

Trois stratégies de fusion, du plus simple au plus riche :
  fusion_stacking  fusion TARDIVE : une régression logistique combine les 4 probabilités
                   (URL, HTML, image, et un XGBoost sur les features techniques). Elle apprend
                   UNIQUEMENT sur les pages de validation, qu'aucun encodeur n'a vues : ses poids
                   ne sont donc pas faussés par le sur-apprentissage des encodeurs. 5 paramètres seulement.
  fusion_concat    fusion INTERMÉDIAIRE : chaque modalité est projetée en 64 nombres, on concatène,
                   puis un petit réseau (MLP) décide. Apprend sur 'train', arrêt anticipé sur 'val'.
  fusion_gated     comme concat, mais une "porte" (attention) donne un poids à chaque modalité pour
                   chaque page : on sait ensuite sur quelle modalité le modèle s'est appuyé (tâche 12).
  Les deux MLP utilisent le "modality dropout" (une modalité entière est masquée au hasard pendant
  l'entraînement) pour ne pas dépendre d'une seule modalité, et une moyenne sur 3 graines.

Ablation (réponse directe à RQ1) : stacking et concat sont refaits sur TOUTES les combinaisons de
modalités (15 combinaisons) -> results/fusion_ablation.csv.

Sorties :
  results/deep_results.csv            lignes tech_xgb, fusion_stacking, fusion_concat, fusion_gated
  results/preds_<modele>.parquet      probabilités des pages de test
  results/fusion_ablation.csv         F1 / PR-AUC de chaque combinaison de modalités
  results/fusion_poids.csv            poids des modalités (coefficients du stacking, portes du gated)
  models/tech_xgb_<proto>.json        XGBoost technique (réutilisé par l'explicabilité, tâche 12)

Usage :
    python src/fusion.py                     # 3 protocoles, toutes les modalités disponibles
    python src/fusion.py --only rq1 --no-ablation
    python src/fusion.py --modalites url html tech      # sans l'image
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_predict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deep_common as dc  # noqa: E402

MODALITIES = ["url", "html", "img", "tech"]
NON_FEATURES = {"sample_id", "split", "label", "chunk"}


# ---------- chargement ------------------------------------------------------------

def tech_matrix(df):
    tech = pd.read_parquet(f"{dc.FEAT}/tech_feat").drop_duplicates("sample_id")
    cols = [c for c in tech.columns if c not in NON_FEATURES]
    X = df[["sample_id"]].merge(tech, on="sample_id", how="left")[cols].astype(float).fillna(0.0)
    return X, cols


def load_modalities(df, proto, wanted):
    """dict modalité -> (X embeddings ou features, p probabilité ou None, présent 0/1), alignés sur df."""
    mods = {}
    for m in wanted:
        if m == "tech":
            X, _ = tech_matrix(df)
            mods["tech"] = (X.to_numpy(np.float32), None, np.ones(len(df), np.float32))
            continue
        path = f"{dc.EMB}/{m}_{proto}.parquet"
        if not os.path.exists(path):
            print(f"  [!] {path} absent : modalité {m} ignorée (lancez d'abord la tâche correspondante)")
            continue
        e = df[["sample_id"]].merge(pd.read_parquet(path), on="sample_id", how="left")
        cols = [c for c in e.columns if c[0] in "uhi" and c[1:].isdigit()]
        present = e["img_ok"].fillna(0).to_numpy(np.float32) if "img_ok" in e else e.p_phishing.notna().to_numpy(np.float32)
        mods[m] = (e[cols].fillna(0).to_numpy(np.float32), e.p_phishing.fillna(0.5).to_numpy(np.float32), present)
    return mods


def scale_tech(X, tr):
    """log(1 + x) pour écraser les grandes valeurs (tailles, nombres de liens), puis standardisation (stats du train)."""
    X = np.sign(X) * np.log1p(np.abs(X))
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    return ((X - mu) / sd).astype(np.float32)


def tech_xgb(Xraw, y, tr, va, proto, seed=42):
    from xgboost import XGBClassifier
    clf = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                        eval_metric="aucpr", early_stopping_rounds=30, random_state=seed, n_jobs=4)
    # arrêt anticipé sur 10 % du train (et pas sur 'val' : 'val' doit rester neuve pour le stacking)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(tr)
    es, fit_idx = perm[:len(perm) // 10], perm[len(perm) // 10:]
    clf.fit(Xraw[fit_idx], y[fit_idx], eval_set=[(Xraw[es], y[es])], verbose=False)
    clf.save_model(f"models/tech_xgb_{proto}.json")
    return clf.predict_proba(Xraw)[:, 1].astype(np.float32)


# ---------- fusion tardive (stacking) ---------------------------------------------

def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def stack_features(mods, names):
    cols, labels = [], []
    for m in names:
        _, p, present = mods[m]
        cols.append(logit(p) * present)                       # image absente -> 0 (aucune information)
        labels.append(m)
        if present.min() == 0:
            cols.append(present)
            labels.append(f"{m}_present")
    return np.column_stack(cols), labels


def stacking(mods, names, y, va, seed=42):
    Z, labels = stack_features(mods, names)
    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(Z[va], y[va])
    # probas "hors pli" sur la validation : sert seulement à choisir un seuil sans tricher
    p_va = cross_val_predict(LogisticRegression(C=1.0, max_iter=1000), Z[va], y[va], cv=5, method="predict_proba")[:, 1]
    lr.labels_ = labels
    return lr.predict_proba(Z)[:, 1], p_va, lr


# ---------- fusion intermédiaire (MLP) --------------------------------------------

class FusionNet(nn.Module):
    def __init__(self, dims, gated, hid=64, drop=0.3, mod_drop=0.25):
        super().__init__()
        self.gated, self.mod_drop = gated, mod_drop
        self.proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Dropout(drop), nn.Linear(d, hid), nn.ReLU())
                                   for d in dims])
        self.score = nn.Linear(hid, 1) if gated else None
        width = hid if gated else hid * len(dims)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(width, hid), nn.ReLU(), nn.Linear(hid, 1))

    def forward(self, xs, masks):
        if self.training and self.mod_drop > 0 and len(xs) > 1:
            keep = (torch.rand_like(masks) > self.mod_drop).float() * masks
            empty = keep.sum(1, keepdim=True) == 0                 # jamais toutes les modalités masquées
            masks = torch.where(empty, masks, keep)
        zs = [p(x) * masks[:, j:j + 1] for j, (p, x) in enumerate(zip(self.proj, xs))]
        if self.gated:
            s = torch.cat([self.score(z) for z in zs], 1).masked_fill(masks == 0, -1e4)
            g = torch.softmax(s, 1)
            h = sum(g[:, j:j + 1] * z for j, z in enumerate(zs))
            return self.head(h).squeeze(1), g
        return self.head(torch.cat(zs, 1)).squeeze(1), None


def train_mlp(Xs, M, y, tr, va, gated, seed, device, epochs=150, patience=12, batch=128):
    dc.seed_all(seed)
    Xs = [torch.tensor(x, device=device) for x in Xs]
    M = torch.tensor(M, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    net = FusionNet([x.shape[1] for x in Xs], gated).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    stop = dc.EarlyStopper(patience)
    rng = np.random.default_rng(seed)
    tr_t = torch.tensor(tr, device=device)
    for ep in range(epochs):
        net.train()
        for b in np.array_split(rng.permutation(len(tr)), max(1, len(tr) // batch)):
            idx = tr_t[torch.tensor(b, device=device)]
            opt.zero_grad()
            out, _ = net([x[idx] for x in Xs], M[idx])
            loss_fn(out, yt[idx]).backward()
            opt.step()
        p_va, _ = run_mlp(net, Xs, M, va)
        if stop.step(average_precision_score(y[va], p_va), net, ep):
            break
    net.load_state_dict(stop.state)
    return net, Xs, M


@torch.no_grad()
def run_mlp(net, Xs, M, idx):
    net.eval()
    idx = torch.as_tensor(idx, device=M.device)
    out, g = net([x[idx] for x in Xs], M[idx])
    return torch.sigmoid(out).cpu().numpy(), (g.cpu().numpy() if g is not None else None)


def mlp_fusion(mods, names, tech_scaled, y, tr, va, gated, seeds, device):
    """Moyenne des probabilités de plusieurs graines (petit ensemble = résultats plus stables)."""
    Xs = [tech_scaled if m == "tech" else mods[m][0] for m in names]
    M = np.column_stack([mods[m][2] for m in names]).astype(np.float32)
    every = np.arange(len(y))
    ps, gs, f1s = [], [], []
    for s in seeds:
        net, Xt, Mt = train_mlp(Xs, M, y, tr, va, gated, s, device)
        p, g = run_mlp(net, Xt, Mt, every)
        ps.append(p)
        if g is not None:
            gs.append(g)
    return np.mean(ps, 0), (np.mean(gs, 0) if gs else None), ps


# ---------- programme principal ---------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=dc.PROTOCOLS)
    ap.add_argument("--modalites", nargs="+", default=MODALITIES, choices=MODALITIES)
    ap.add_argument("--seeds", type=int, default=3, help="graines par MLP (moyenne)")
    ap.add_argument("--no-ablation", action="store_true", help="ne pas tester toutes les combinaisons")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = dc.device_of(args.cpu)
    dc.setup_dirs()
    df = dc.load_index()
    y = df.y.to_numpy()
    Xtech_raw, tech_cols = tech_matrix(df)
    Xtech_raw = Xtech_raw.to_numpy(np.float32)
    seeds = list(range(42, 42 + args.seeds))

    rows, preds, ablation, poids = [], {}, [], []
    for proto in dc.protocols_arg(args.only):
        part = dc.protocol_parts(df, proto)
        tr, va, te = (np.flatnonzero(part == s) for s in ("train", "val", "test"))
        print(f"\n=== {proto} : {len(tr)} train, {len(va)} validation, {len(te)} test ===")
        mods = load_modalities(df, proto, args.modalites)
        if "tech" in mods:                                    # XGBoost technique entraîné sur 'train' seulement
            p_tech = tech_xgb(Xtech_raw, y, tr, va, proto)
            mods["tech"] = (mods["tech"][0], p_tech, mods["tech"][2])
        names = [m for m in MODALITIES if m in mods]
        tech_scaled = scale_tech(Xtech_raw, tr)
        print("  modalités :", ", ".join(f"{m} ({mods[m][0].shape[1]} dims)" for m in names))

        def record(model, p, p_va):
            thr = dc.best_threshold(y[va], p_va)
            m = dc.metrics(y[te], p[te], thr)
            rows.append({"protocole": proto, "modele": model, **m, "modalites": "+".join(names), "appareil": device})
            preds.setdefault(model, []).append(pd.DataFrame({"sample_id": df.sample_id.to_numpy()[te],
                                                             "protocole": proto, "modele": model,
                                                             "y": y[te], "p_phishing": p[te]}))
            print(f"  {model:16s}", dc.fmt(m))
            return m

        if "tech" in mods:
            record("tech_xgb", mods["tech"][1], mods["tech"][1][va])

        # 1. stacking sur les 4 probabilités
        with dc.Timer() as t:
            p, p_va, lr = stacking(mods, names, y, va)
        record("fusion_stacking", p, p_va)
        coefs = dict(zip(lr.labels_, lr.coef_[0]))
        for n in names:
            poids.append({"protocole": proto, "strategie": "stacking (coefficient)", "modalite": n,
                          "poids": round(float(coefs[n]), 4)})

        # 2. et 3. MLP concat et gated
        for gated, model in [(False, "fusion_concat"), (True, "fusion_gated")]:
            with dc.Timer() as t:
                p, g, runs = mlp_fusion(mods, names, tech_scaled, y, tr, va, gated, seeds, device)
            m = record(model, p, p[va])
            rows[-1]["train_s"] = round(t.s, 1)
            f1_runs = [dc.metrics(y[te], r[te])["f1"] for r in runs]
            rows[-1]["f1_graines_std"] = float(np.std(f1_runs))
            if g is not None:
                for j, n in enumerate(names):
                    for cls, lab in [(0, "légitime"), (1, "phishing")]:
                        sel = te[y[te] == cls]
                        poids.append({"protocole": proto, "strategie": f"gated (porte moyenne, {lab})",
                                      "modalite": n, "poids": round(float(g[sel, j].mean()), 4)})
                gdf = pd.DataFrame(g, columns=[f"porte_{n}" for n in names])
                gdf.insert(0, "sample_id", df.sample_id.to_numpy())
                gdf.iloc[te].to_parquet(f"results/fusion_portes_{proto}.parquet", index=False)

        # 4. ablation : toutes les combinaisons de modalités
        if not args.no_ablation and len(names) > 1:
            combos = [c for k in range(1, len(names) + 1) for c in itertools.combinations(names, k)]
            print(f"  ablation : {len(combos)} combinaisons x 2 stratégies")
            for c in combos:
                c = list(c)
                p_s, p_s_va, _ = stacking(mods, c, y, va)
                p_c, _, _ = mlp_fusion(mods, c, tech_scaled, y, tr, va, False, seeds[:2], device)
                for strat, pp in [("stacking", p_s), ("concat", p_c)]:
                    mm = dc.metrics(y[te], pp[te])
                    ablation.append({"protocole": proto, "strategie": strat, "modalites": "+".join(c),
                                     "n_modalites": len(c), **{k: mm[k] for k in ["f1", "precision", "recall",
                                                                                  "roc_auc", "pr_auc"]}})
            ab = pd.DataFrame([a for a in ablation if a["protocole"] == proto])
            print(ab.sort_values("pr_auc", ascending=False).head(8).round(3).to_string(index=False))

    dc.save_results(rows)
    for model, frames in preds.items():
        dc.save_preds(frames, model)
    if ablation:
        path = "results/fusion_ablation.csv"
        ab = pd.DataFrame(ablation)
        if os.path.exists(path):
            old = pd.read_csv(path)
            ab = pd.concat([old[~old.protocole.isin(ab.protocole)], ab])
        ab.round(4).to_csv(path, index=False)
    pd.DataFrame(poids).to_csv("results/fusion_poids.csv", index=False)

    print("\nRésumé fusion (results/deep_results.csv) :")
    print(pd.DataFrame(rows)[["protocole", "modele", "f1", "precision", "recall", "roc_auc", "pr_auc", "f1_seuil_val"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
