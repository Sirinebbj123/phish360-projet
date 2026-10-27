"""Tâche 3 (3/3) - décide quelles pages garder, et pourquoi.

On ne réécrit PAS les 4 Go de HTML et d'images : on produit un petit index
(data/clean_index.parquet) avec keep = True/False et la raison. Les tâches suivantes
feront une jointure raw + index pour ne lire que les pages gardées.

Règles (dans cet ordre) :
  1. modalité manquante (HTML ou image)  -> retirée : la fusion a besoin des 4 modalités (RQ1)
  2. URL en double dans un même split    -> on garde la première, les copies sont retirées
  3. URL du test déjà présente dans trainval -> retirée du test (fuite de données)

Usage :  python src/clean_index.py
"""
import pandas as pd

meta = pd.read_parquet("data/phish360_meta.parquet")
raw_missing = pd.read_parquet("data/phish360_raw.parquet", columns=["sample_id", "split", "missing"])
df = meta.merge(raw_missing, on=["sample_id", "split"])

# URL normalisée : minuscules, sans "/" final ni espaces
df["url_norm"] = df.url.str.strip().str.lower().str.rstrip("/")
df = df.sort_values(["split", "sample_id"]).reset_index(drop=True)
df["reason"] = None

has_missing = df.missing.apply(lambda m: m is not None and len(m) > 0) | df.img_w.isna()
df.loc[has_missing, "reason"] = "modalite_manquante"

dup = df.reason.isna() & df.duplicated(["split", "url_norm"], keep="first")
df.loc[dup, "reason"] = "doublon_meme_split"

train_urls = set(df.loc[(df.split == "trainval") & df.reason.isna(), "url_norm"])
leak = df.reason.isna() & (df.split == "test") & df.url_norm.isin(train_urls)
df.loc[leak, "reason"] = "fuite_train_test"

df["keep"] = df.reason.isna()
print("Pages retirées par raison :")
print(df.reason.value_counts().to_string())
print("\nPages gardées par split et label :")
print(df[df.keep].groupby(["split", "label"]).size().unstack().to_string())

cols = ["sample_id", "split", "label", "brand", "keep", "reason", "url_norm", "host",
        "id_num", "year_copyright", "year_max"]
df[cols].to_parquet("data/clean_index.parquet", index=False)
print(f"\n{df.keep.sum()} pages gardées sur {len(df)} -> data/clean_index.parquet")
