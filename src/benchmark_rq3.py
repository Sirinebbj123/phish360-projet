"""Tâche 11 - benchmark RQ3 : coût du pipeline distribué selon le volume et le nombre de cœurs.

Chaque "run" relance le vrai pipeline sur une partie des données :
  1. prétraitement URL + HTML + image (mêmes fonctions que src/preprocess.py)
  2. entraînement d'un GBT MLlib sur les features techniques (comme tech_gbt en tâche 5)

Le volume est réglé en prenant une partie des lots du Parquet brut (25 % = 6 lots sur 22, etc.).
Par défaut 5 runs (environ 30 à 40 minutes avec les pauses) :
  - volume : 25 %, 50 %, 100 % avec 4 cœurs         -> le temps suit-il le volume ? (passage à l'échelle)
  - cœurs  : 1, 2, 4 avec 50 % du volume            -> combien gagne-t-on en ajoutant des cœurs ? (accélération)

Sorties :
  results/rq3_benchmark.csv
  results/figures/rq3_volume.png   temps selon le volume
  results/figures/rq3_coeurs.png   temps et accélération selon le nombre de cœurs

Complément Sprint 2 :
  --grid      grille complète volumes x cœurs (9 runs au lieu de 5)
  --repeat 3  chaque configuration répétée 3 fois -> moyenne et écart-type (figures avec barres d'erreur)

Usage :
    python src/benchmark_rq3.py
    python src/benchmark_rq3.py --grid --repeat 3 --out results/rq3_benchmark_grille.csv   # ≈ 2 h
    python src/benchmark_rq3.py --volumes 0.25 0.5 --cores 1 2      # version plus courte
"""
import argparse
import os
import shutil
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preprocess  # noqa: E402  (mêmes fonctions que la tâche 4)

RAW, INDEX, TMP = "data/phish360_raw.parquet", "data/clean_index.parquet", "data/bench_tmp"
NON_FEATURES = {"sample_id", "split", "label", "chunk"}


def run(cores, volume, memory):
    spark = (SparkSession.builder.master(f"local[{cores}]").appName(f"rq3_{cores}c_{volume}")
             .config("spark.driver.memory", memory)
             .config("spark.sql.shuffle.partitions", str(cores * 4))
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    # les cœurs Spark doivent pouvoir importer preprocess.py
    spark.sparkContext.addPyFile(os.path.abspath(preprocess.__file__))

    chunks = sorted(d for d in os.listdir(RAW) if d.startswith("chunk="))
    n_chunks = max(1, round(len(chunks) * volume))
    paths = [os.path.join(RAW, c) for c in chunks[:n_chunks]]
    keep = spark.read.parquet(INDEX).where("keep").select("sample_id", "split")
    res = {"coeurs": cores, "volume": volume, "lots": n_chunks}

    # 1. prétraitement des 3 modalités
    for step, (cols, fn, schema) in preprocess.STEPS.items():
        t0 = time.perf_counter()
        raw = spark.read.parquet(*paths).select("sample_id", "split", "label", *cols)
        df = raw.join(F.broadcast(keep), ["sample_id", "split"])
        out = os.path.join(TMP, step)
        spark.createDataFrame(df.rdd.map(fn), schema).write.mode("overwrite").parquet(out)
        res[f"{step}_s"] = round(time.perf_counter() - t0, 1)
        if step == "url":
            res["pages"] = spark.read.parquet(out).count()

    # 2. entraînement GBT sur les features techniques des mêmes pages
    ids = spark.read.parquet(os.path.join(TMP, "url")).select("sample_id")
    tech = spark.read.parquet("data/features/tech_feat").join(ids, "sample_id")
    cols = [c for c in tech.columns if c not in NON_FEATURES]
    tech = tech.fillna(0, subset=cols).withColumn("y", (F.col("label") == "phishing").cast("double"))
    t0 = time.perf_counter()
    Pipeline(stages=[VectorAssembler(inputCols=cols, outputCol="f"),
                     GBTClassifier(featuresCol="f", labelCol="y", maxIter=50, maxDepth=5, seed=42)]).fit(tech)
    res["gbt_s"] = round(time.perf_counter() - t0, 1)

    res["pretraitement_s"] = round(res["url_s"] + res["html_s"] + res["image_s"], 1)
    res["total_s"] = round(res["pretraitement_s"] + res["gbt_s"], 1)
    res["pages_par_s"] = round(res["pages"] / res["pretraitement_s"], 1)
    spark.stop()
    shutil.rmtree(TMP, ignore_errors=True)
    return res


def plots(raw):
    os.makedirs("results/figures", exist_ok=True)
    # plusieurs répétitions d'une même configuration -> moyenne (et écart-type pour les barres d'erreur)
    keys = ["coeurs", "volume"]
    num = [c for c in raw.columns if c.endswith("_s") or c in ("pages", "pages_par_s")]
    df = raw.groupby(keys, as_index=False)[num].mean()
    sd = raw.groupby(keys, as_index=False)[num].std().fillna(0)
    vol = df[df.coeurs == df.coeurs.max()].sort_values("pages")
    if len(vol) > 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        for col, lab in [("url_s", "URL"), ("html_s", "HTML"), ("image_s", "image"), ("gbt_s", "GBT"), ("total_s", "total")]:
            ax.plot(vol.pages, vol[col], marker="o", label=lab, lw=2.5 if col == "total_s" else 1.5)
        ax.set_xlabel("pages traitées"); ax.set_ylabel("secondes")
        ax.set_title(f"RQ3 : temps selon le volume ({int(vol.coeurs.iloc[0])} cœurs)")
        ax.legend(frameon=False); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig("results/figures/rq3_volume.png", dpi=150); plt.close(fig)

    v_mid = df.groupby("volume").coeurs.nunique().idxmax()
    cor = df[df.volume == v_mid].sort_values("coeurs")
    if len(cor) > 1:
        base = cor.total_s.iloc[0] * cor.coeurs.iloc[0]          # temps ramené à 1 cœur
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
        err = sd.set_index(keys).loc[list(zip(cor.coeurs, cor.volume)), "total_s"].to_numpy()
        a1.bar(cor.coeurs.astype(str), cor.total_s, color="#3a6ea5", yerr=err, capsize=3)
        a1.set_xlabel("cœurs"); a1.set_ylabel("secondes"); a1.set_title(f"Temps total ({int(v_mid * 100)} % du volume)")
        a2.plot(cor.coeurs, base / cor.total_s, marker="o", label="mesurée")
        a2.plot(cor.coeurs, cor.coeurs, ls="--", color="gray", label="idéale")
        a2.set_xlabel("cœurs"); a2.set_ylabel("accélération (× plus rapide)"); a2.set_title("Accélération")
        a2.legend(frameon=False); a2.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig("results/figures/rq3_coeurs.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volumes", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--cores", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--memory", default="4g")
    ap.add_argument("--pause", type=int, default=60, help="secondes de pause entre deux runs (refroidissement)")
    ap.add_argument("--grid", action="store_true", help="toutes les combinaisons volumes x cœurs")
    ap.add_argument("--repeat", type=int, default=1, help="répétitions de chaque configuration")
    ap.add_argument("--out", default="results/rq3_benchmark.csv")
    args = ap.parse_args()

    max_c, mid_v = max(args.cores), sorted(args.volumes)[len(args.volumes) // 2]
    if args.grid:
        plan = [(c, v) for c in sorted(args.cores) for v in sorted(args.volumes)]
    else:
        plan = [(max_c, v) for v in sorted(args.volumes)] + [(c, mid_v) for c in sorted(args.cores) if c != max_c]
    plan = [cv for cv in plan for _ in range(args.repeat)]
    print(f"{len(plan)} runs prévus :", ", ".join(f"{c} cœurs / {int(v * 100)} %" for c, v in plan))

    out = args.out
    rows = []
    for i, (c, v) in enumerate(plan, 1):
        print(f"\n--- run {i}/{len(plan)} : {c} cœurs, {int(v * 100)} % du volume ---")
        r = run(c, v, args.memory)
        r["repetition"] = sum(1 for x in rows if x["coeurs"] == c and x["volume"] == v) + 1
        rows.append(r)
        print(f"{r['pages']} pages : URL {r['url_s']} s, HTML {r['html_s']} s, image {r['image_s']} s, "
              f"GBT {r['gbt_s']} s -> total {r['total_s']} s")
        pd.DataFrame(rows).to_csv(out, index=False)            # sauvegarde après chaque run
        if args.pause and i < len(plan):
            time.sleep(args.pause)

    df = pd.DataFrame(rows)
    plots(df)
    print(f"\nRésumé ({out}, moyenne des répétitions) :")
    cols = ["pages", "url_s", "html_s", "image_s", "gbt_s", "total_s", "pages_par_s"]
    summary = df.groupby(["coeurs", "volume"])[cols].agg(["mean", "std"]) if args.repeat > 1 else df[["coeurs", "volume"] + cols]
    print(summary.round(1).to_string())
    print("Figures : results/figures/rq3_volume.png et rq3_coeurs.png")


if __name__ == "__main__":
    main()
