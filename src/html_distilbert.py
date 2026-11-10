"""Tâche 7 - modèle deep sur le texte HTML : DistilBERT affiné (fine-tuning).

Entrée : data/features/html_text (tâche 4) = titre + texte visible (4 000 caractères, années masquées).
Le texte est découpé en tokens (sous-mots) puis passe dans DistilBERT (6 couches de Transformer,
pré-entraîné sur Wikipédia). Le vecteur du token [CLS] résume la page ; une petite tête
(Dropout -> 256 -> ReLU -> 1) donne le score "phishing". Tout le réseau est réentraîné (fine-tuning).

Modèle par défaut : distilbert-base-multilingual-cased (les pages sont en anglais, mais aussi en
tchèque, allemand, roumain, portugais, russe... cf. EDA). Variante anglaise : --model distilbert-base-uncased.

Réglages pour une RTX 3050 6 Go : 256 tokens, lots de 16, précision mixte (fp16). En cas de
"CUDA out of memory" : --batch 8 --accum 2 (même lot effectif de 16).

Sorties :
  results/deep_results.csv              une ligne par protocole (modele = html_distilbert)
  results/preds_html_distilbert.parquet probabilité "phishing" des pages de test
  models/html_distilbert_<proto>.pt     poids (≈ 540 Mo chacun, ignorés par Git)
  data/emb/html_<proto>.parquet         256 nombres + probabilité par page (entrée de la fusion, tâche 9)

Usage :
    python src/html_distilbert.py                        # 3 protocoles (≈ 15-25 min sur RTX 3050)
    python src/html_distilbert.py --only rq1 --epochs 2  # essai rapide
    python src/html_distilbert.py --freeze               # DistilBERT figé, seule la tête apprend (CPU possible)
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deep_common as dc  # noqa: E402

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")   # avertissement inutile sous Windows
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
DEFAULT_MODEL = "distilbert-base-multilingual-cased"
EMB_DIM = 256
AMP = True          # précision mixte fp16 (désactivable avec --no-amp)


class HtmlClassifier(nn.Module):
    def __init__(self, model_name, dropout=0.2):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(model_name)
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.bert.config.hidden_size, EMB_DIM), nn.ReLU())
        self.head = nn.Linear(EMB_DIM, 1)

    def cls(self, ids, mask):
        return self.bert(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]   # vecteur [CLS]

    def embed(self, ids, mask):
        return self.proj(self.cls(ids, mask))

    def forward(self, ids, mask):
        return self.head(self.embed(ids, mask)).squeeze(1)


def load_texts(df):
    html = pd.read_parquet(f"{dc.FEAT}/html_text", columns=["sample_id", "title", "text"]).drop_duplicates("sample_id")
    d = df.merge(html, on="sample_id", how="left")
    title, text = d.title.fillna("").str.strip(), d.text.fillna("").str.strip()
    # titre en premier : il est court et très informatif ; "[vide]" si la page n'a aucun texte
    doc = (title + " | " + text).str.strip(" |")
    return doc.where(doc.str.len() > 0, "[vide]").tolist()


def tokenize(tokenizer, docs, max_len):
    enc = tokenizer(docs, truncation=True, max_length=max_len, padding=False)
    return enc["input_ids"]


def make_batch(ids_list, idx, pad_id, device):
    """Remplissage dynamique : chaque lot est complété à la longueur de son plus long texte (plus rapide)."""
    L = max(len(ids_list[i]) for i in idx)
    ids = torch.full((len(idx), L), pad_id, dtype=torch.long)
    for r, i in enumerate(idx):
        ids[r, :len(ids_list[i])] = torch.tensor(ids_list[i])
    mask = (ids != pad_id).long()
    return ids.to(device, non_blocking=True), mask.to(device, non_blocking=True)


@torch.no_grad()
def predict(model, ids_list, idx_all, pad_id, device, batch=64, amp=True):
    """Probabilités et vecteurs (256) pour les pages idx_all. Les textes sont triés par longueur (moins de remplissage)."""
    model.eval()
    order = sorted(idx_all, key=lambda i: len(ids_list[i]))
    ps, es = {}, {}
    for k in range(0, len(order), batch):
        chunk = order[k:k + batch]
        ids, mask = make_batch(ids_list, chunk, pad_id, device)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp and AMP and device == "cuda"):
            e = model.embed(ids, mask)
            p = torch.sigmoid(model.head(e).squeeze(1))
        for i, pi, ei in zip(chunk, p.float().cpu().numpy(), e.float().cpu().numpy()):
            ps[i], es[i] = pi, ei
    return np.array([ps[i] for i in idx_all]), np.stack([es[i] for i in idx_all])


def train_full(model, ids_list, y, tr, va, pad_id, args, device):
    """Fine-tuning complet : AdamW, lr 2e-5 pour DistilBERT et 1e-3 pour la tête, échauffement + décroissance linéaire."""
    from transformers import get_linear_schedule_with_warmup
    if not args.train_emb:
        # la table des mots (119 000 x 768 = 92 M de paramètres, 70 % du modèle multilingue) reste figée :
        # gradients + états d'AdamW en moins = environ 1,5 Go de mémoire GPU en moins, et beaucoup plus rapide
        # sur les cartes de 4 Go (sinon Windows déborde en mémoire partagée, très lente). Effet sur le score : minime.
        model.bert.embeddings.word_embeddings.weight.requires_grad = False
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  paramètres entraînés : {n_train / 1e6:.0f} M sur {sum(p.numel() for p in model.parameters()) / 1e6:.0f} M")
    opt = torch.optim.AdamW([{"params": [p for p in model.bert.parameters() if p.requires_grad], "lr": args.lr},
                             {"params": list(model.proj.parameters()) + list(model.head.parameters()), "lr": 1e-3}],
                            weight_decay=0.01)
    steps = math.ceil(len(tr) / (args.batch * args.accum)) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP and device == "cuda")
    loss_fn = nn.BCEWithLogitsLoss()
    stop = dc.EarlyStopper(args.patience)
    rng = np.random.default_rng(42)
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, total, n = time.perf_counter(), 0.0, 0
        perm = rng.permutation(tr)
        opt.zero_grad()
        for k in range(0, len(perm), args.batch):
            idx = perm[k:k + args.batch]
            ids, mask = make_batch(ids_list, idx, pad_id, device)
            yb = torch.tensor(y[idx], dtype=torch.float32, device=device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=AMP and device == "cuda"):
                loss = loss_fn(model(ids, mask), yb)
            scaler.scale(loss / args.accum).backward()
            if (k // args.batch + 1) % args.accum == 0 or k + args.batch >= len(perm):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sched.step()
            total, n = total + loss.item() * len(idx), n + len(idx)
            if (k // args.batch) % 100 == 0:
                print(f"    lot {k // args.batch:4d}/{math.ceil(len(perm) / args.batch)}  perte {total / n:.4f}  "
                      f"({time.perf_counter() - t0:.0f} s écoulées)", flush=True)
        p_val, _ = predict(model, ids_list, list(va), pad_id, device)
        if not np.isfinite(p_val).all():
            raise SystemExit("Le réseau produit des NaN : relancez avec --no-amp (calcul en fp32), et si besoin --lr 1e-5.")
        score = average_precision_score(y[va], p_val)
        print(f"  époque {ep} : perte {total / n:.4f}  PR-AUC validation {score:.4f}  ({time.perf_counter() - t0:.0f} s)")
        if stop.step(score, model, ep):
            print("  arrêt anticipé")
            break
        if args.pause:
            time.sleep(args.pause)
    model.load_state_dict(stop.state)
    print(f"  meilleure époque : {stop.epoch} (PR-AUC validation {stop.best:.4f})")
    return model


def train_frozen(model, ids_list, y, tr, va, used, pad_id, args, device):
    """DistilBERT figé : on calcule une fois le vecteur [CLS] de chaque page, puis seule la tête apprend."""
    model.eval()
    order = sorted(used, key=lambda i: len(ids_list[i]))
    cls = {}
    with torch.no_grad():
        for k in range(0, len(order), 64):
            chunk = order[k:k + 64]
            ids, mask = make_batch(ids_list, chunk, pad_id, device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=AMP and device == "cuda"):
                c = model.cls(ids, mask).float().cpu()
            for i, ci in zip(chunk, c):
                cls[i] = ci
            if k % 1280 == 0:
                print(f"    [CLS] {k}/{len(order)} pages", flush=True)
    C = torch.zeros(len(y), model.bert.config.hidden_size)
    for i, c in cls.items():
        C[i] = c
    head = nn.Sequential(model.proj, model.head).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    stop = dc.EarlyStopper(args.patience + 3)
    rng = np.random.default_rng(42)
    for ep in range(1, 61):
        head.train()
        for idx in np.array_split(rng.permutation(tr), max(1, len(tr) // 64)):
            opt.zero_grad()
            loss = loss_fn(head(C[idx].to(device)).squeeze(1), torch.tensor(y[idx], dtype=torch.float32, device=device))
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            p_val = torch.sigmoid(head(C[va].to(device)).squeeze(1)).cpu().numpy()
        score = average_precision_score(y[va], p_val)
        if stop.step(score, head, ep):
            break
    head.load_state_dict(stop.state)
    print(f"  tête entraînée : meilleure époque {stop.epoch} (PR-AUC validation {stop.best:.4f})")
    head.eval()
    with torch.no_grad():
        e = model.proj(C[used].to(device))
        p = torch.sigmoid(model.head(e).squeeze(1))
    return p.cpu().numpy(), e.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=dc.PROTOCOLS)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="nom Hugging Face ou dossier local")
    ap.add_argument("--max-len", type=int, default=256, help="tokens lus par page (512 = plus lent, +mémoire)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1, help="accumulation de gradient (lot effectif = batch x accum)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--freeze", action="store_true", help="DistilBERT figé (rapide, possible sur CPU)")
    ap.add_argument("--pause", type=int, default=10, help="secondes de pause entre deux époques")
    ap.add_argument("--no-save-model", action="store_true", help="ne pas écrire models/*.pt (gain de place)")
    ap.add_argument("--train-emb", action="store_true",
                    help="réentraîner aussi la table des mots (+1,5 Go de mémoire ; inutile sur une carte de 4 Go)")
    ap.add_argument("--no-amp", action="store_true", help="désactiver la précision mixte fp16 (calcul en fp32, + mémoire)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    global AMP
    AMP = not args.no_amp
    device = dc.device_of(args.cpu)
    dc.setup_dirs()
    name = "html_distilbert_gele" if args.freeze else "html_distilbert"

    df = dc.load_index()
    docs = load_texts(df)
    tok = AutoTokenizer.from_pretrained(args.model)
    t0 = time.perf_counter()
    ids_list = tokenize(tok, docs, args.max_len)
    lens = np.array([len(x) for x in ids_list])
    print(f"{len(docs)} textes découpés en tokens en {time.perf_counter() - t0:.0f} s ; "
          f"longueur médiane {int(np.median(lens))}, {np.mean(lens >= args.max_len):.0%} tronqués à {args.max_len}")
    y = df.y.to_numpy()

    rows, preds = [], []
    for proto in dc.protocols_arg(args.only):
        part = dc.protocol_parts(df, proto)
        tr, va, te = (np.flatnonzero(part == s) for s in ("train", "val", "test"))
        used = np.flatnonzero(part != "hors")
        print(f"\n=== {proto} : {len(tr)} pages d'entraînement, {len(va)} de validation, {len(te)} de test ===")
        dc.seed_all(42)
        model = HtmlClassifier(args.model).to(device)
        with dc.Timer() as t:
            if args.freeze:
                p_used, e_used = train_frozen(model, ids_list, y, tr, va, used, tok.pad_token_id, args, device)
            else:
                model = train_full(model, ids_list, y, tr, va, tok.pad_token_id, args, device)
        with dc.Timer() as ti:
            if not args.freeze:
                p_used, e_used = predict(model, ids_list, list(used), tok.pad_token_id, device)
        p = np.full(len(df), np.nan)
        p[used] = p_used
        thr = dc.best_threshold(y[va], p[va])
        m = dc.metrics(y[te], p[te], thr)
        rows.append({"protocole": proto, "modele": name, **m, "train_s": round(t.s, 1), "appareil": device,
                     "inference_pages_s": round(len(used) / ti.s, 1) if ti.s > 0.01 else None})
        print(name, dc.fmt(m), f"({t.s:.0f} s)")
        dc.log_time(f"deep_{proto}_{name}", len(tr), t.s, device)
        preds.append(pd.DataFrame({"sample_id": df.sample_id.to_numpy()[te], "protocole": proto, "modele": name,
                                   "y": y[te], "p_phishing": p[te]}))
        if not args.no_save_model:
            torch.save({"model_name": args.model, "max_len": args.max_len, "state_dict": model.state_dict()},
                       f"models/{name}_{proto}.pt")
        dc.save_embeddings(df.iloc[used], part[used], e_used, p_used, "html", proto, "h")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    dc.save_results(rows)
    dc.save_preds(preds, name)
    print("\nComparaison avec la baseline HTML TF-IDF (tâche 5) :")
    res = pd.DataFrame(rows)
    if os.path.exists("results/baselines.csv"):
        res = pd.concat([pd.read_csv("results/baselines.csv").query("modele == 'html_tfidf'"), res])
    print(res[["protocole", "modele", "f1", "precision", "recall", "pr_auc"]]
          .sort_values(["protocole", "modele"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
