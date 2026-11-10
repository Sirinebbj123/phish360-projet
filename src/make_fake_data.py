"""Données FACTICES pour tester les scripts du Sprint 2 sans le vrai dataset (ni GPU).

Crée, dans un dossier à part (jamais dans data/ du projet), la même structure que les tâches 3 à 5 :
  <dossier>/data/features/{url_feat,html_text,img_224,tech_feat}/chunk=000/part-0.parquet
  <dossier>/data/rq2_split.parquet
  <dossier>/results/baselines.csv et preds_baselines.parquet (fausses baselines)
Les pages "phishing" ont des URL, textes, images et features un peu différents : les modèles
doivent donc apprendre quelque chose, mais les scores n'ont AUCUN sens scientifique.

Usage :
    python src/make_fake_data.py demo_factice --pages 600
    cd demo_factice
    python ../src/url_charcnn.py --epochs 2 --pause 0
"""
import argparse
import io
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

BRANDS = ["paypal", "dhl", "microsoft", "apple", "netflix", "amazon", "bankofamerica", "outlook"]
WORDS_OK = "welcome news products about contact careers blog shop team history support help store".split()
WORDS_PH = "verify account login password suspended confirm security update billing unlock".split()


def write(df, path):
    os.makedirs(path, exist_ok=True)
    df.to_parquet(os.path.join(path, "part-0.parquet"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--pages", type=int, default=600)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    n = args.pages
    y = (rng.random(n) < 0.4).astype(int)
    ids = [f"P{i:05d}_{rng.choice(BRANDS)}" if yi else f"L{i:05d}_legitimate" for i, yi in enumerate(y)]
    split = np.where(rng.random(n) < 0.74, "trainval", "test")
    label = np.where(y == 1, "phishing", "legitimate")
    noisy = rng.random(n) < 0.15                                    # 15 % d'exemples "difficiles"
    sig = np.where(noisy, 1 - y, y)

    urls = [f"http://{rng.choice(['secure-', 'login.', ''])}{rng.choice(BRANDS)}-{rng.integers(100, 999)}.xyz/verify?id={rng.integers(1e6)}"
            if s else f"https://www.{rng.choice(['example', 'news', 'shop', 'univ'])}{rng.integers(10)}.com/{rng.choice(WORDS_OK)}"
            for s in sig]
    base = os.path.join(args.out, "data", "features")
    write(pd.DataFrame({"sample_id": ids, "split": split, "label": label, "url": urls,
                        "suffix": ["xyz" if s else "com" for s in sig]}), f"{base}/url_feat/chunk=000")

    texts = [" ".join(rng.choice(WORDS_PH if s else WORDS_OK, rng.integers(5, 60))) for s in sig]
    texts = [t if rng.random() > 0.05 else "" for t in texts]
    write(pd.DataFrame({"sample_id": ids, "split": split, "label": label,
                        "title": [f"{'Sign in' if s else 'Home'} {rng.choice(BRANDS)}" for s in sig],
                        "text": texts, "text_len": [len(t) for t in texts]}), f"{base}/html_text/chunk=000")

    blobs, ok = [], []
    for s in sig:
        im = Image.new("RGB", (224, 224), tuple(int(c) for c in rng.integers(150, 255, 3)))
        d = ImageDraw.Draw(im)
        if s:                                                       # "formulaire" au centre
            d.rectangle([60, 80, 164, 150], fill=(255, 255, 255), outline=(0, 0, 200), width=3)
        else:
            for k in range(5):
                d.rectangle([10, 10 + 40 * k, 214, 30 + 40 * k], fill=tuple(int(c) for c in rng.integers(0, 150, 3)))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        miss = rng.random() < 0.03
        blobs.append(None if miss else buf.getvalue())
        ok.append(0 if miss else 1)
    write(pd.DataFrame({"sample_id": ids, "split": split, "label": label, "img": blobs, "img_ok": ok}),
          f"{base}/img_224/chunk=000")

    tech = {"sample_id": ids, "split": split, "label": label}
    for j in range(35):
        tech[f"f{j:02d}"] = rng.poisson(5 + 3 * sig * (j % 3 == 0), n).astype(float)
    tech["text_len"] = [len(t) for t in texts]
    write(pd.DataFrame(tech), f"{base}/tech_feat/chunk=000")

    period = np.where(y == 1, np.where(rng.random(n) < 0.6, np.where(rng.random(n) < 0.57, "ancienne", "recente"), None),
                      np.where(rng.random(n) < 0.57, "ancienne", "recente"))
    os.makedirs(os.path.join(args.out, "data"), exist_ok=True)
    pd.DataFrame({"sample_id": ids, "split": split, "label": label, "rq2_period": period}).to_parquet(
        os.path.join(args.out, "data", "rq2_split.parquet"), index=False)

    # fausses baselines (tâche 5) pour tester l'évaluation finale
    os.makedirs(os.path.join(args.out, "results"), exist_ok=True)
    rows, preds = [], []
    for proto, te in [("rq1", split == "test"), ("rq2", period == "recente")]:
        for m, noise in [("url_tfidf", 0.9), ("html_tfidf", 0.6), ("tech_gbt", 0.5), ("concat", 0.5)]:
            p = 1 / (1 + np.exp(-(2 * sig[te] - 1) * 2 + rng.normal(0, noise * 2, te.sum())))
            preds.append(pd.DataFrame({"sample_id": np.array(ids)[te], "protocole": proto, "modele": m,
                                       "y": y[te].astype(float), "p_phishing": p}))
            rows.append({"protocole": proto, "modele": m, "train_s": 10.0})
    pd.concat(preds).to_parquet(os.path.join(args.out, "results", "preds_baselines.parquet"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "results", "baselines.csv"), index=False)
    print(f"{n} pages factices -> {args.out}/data ; lancez les scripts depuis ce dossier.")


if __name__ == "__main__":
    main()
