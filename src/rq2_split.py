"""Tâche 3 - découpage temporel pour RQ2 (option A' : âge des kits de phishing).

Constat de l'EDA : les dates de fichiers sont inutilisables, les numéros de dossier ne suivent
pas le temps, et l'année de copyright des pages légitimes vaut presque toujours 2020
(date de collecte). En revanche, pour les pages de PHISHING, l'année de copyright varie de 2007
à 2021 : elle date le kit / la campagne. RQ2 porte justement sur des campagnes de phishing
postérieures à l'entraînement.

Découpage (sur toutes les pages gardées, trainval + test réunis) :
  - phishing avec année <= SEUIL      -> période "ancienne"  (entraînement RQ2)
  - phishing avec année  > SEUIL      -> période "récente"   (test RQ2)
  - phishing sans année               -> hors expérience RQ2 (gardé pour RQ1)
  - légitimes : répartis au hasard (graine fixe) dans les deux périodes, même proportion
Usage :  python src/rq2_split.py [--seuil 2018]
"""
import argparse

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--seuil", type=int, default=2018)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

idx = pd.read_parquet("data/clean_index.parquet")
df = idx[idx.keep].copy()
phish = df.label == "phishing"
dated = phish & df.year_copyright.notna()

df["rq2_period"] = None
df.loc[dated & (df.year_copyright <= args.seuil), "rq2_period"] = "ancienne"
df.loc[dated & (df.year_copyright > args.seuil), "rq2_period"] = "recente"

# Légitimes : même part "ancienne" que chez les phishing datés
share_old = (df.rq2_period == "ancienne").sum() / dated.sum()
rng = np.random.default_rng(args.seed)
legit = df.label == "legitimate"
df.loc[legit, "rq2_period"] = np.where(rng.random(legit.sum()) < share_old, "ancienne", "recente")

print(f"Seuil : copyright <= {args.seuil} = ancienne ; part ancienne = {share_old:.2f}")
print(pd.crosstab(df.rq2_period.fillna("hors RQ2"), df.label, margins=True).to_string())
print("\nAnnées de copyright des phishing par période :")
print(df[dated].groupby("rq2_period").year_copyright.agg(["min", "median", "max", "count"]).to_string())

df[["sample_id", "split", "label", "year_copyright", "rq2_period"]].to_parquet(
    "data/rq2_split.parquet", index=False)
print("\n-> data/rq2_split.parquet")
