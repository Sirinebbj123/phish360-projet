"""Tâche 5 - baselines classiques avec Spark MLlib.

Quatre modèles, entraînés et évalués en distribué :
  url_tfidf   n-grammes de caractères (3 et 4) de l'URL -> TF-IDF -> régression logistique
  html_tfidf  mots du titre + texte visible -> TF-IDF -> régression logistique
  tech_gbt    35 features techniques -> arbres boostés (GBT)
  concat      les trois vecteurs mis bout à bout -> régression logistique (fusion "naïve")

Deux protocoles :
  rq1  entraînement sur trainval, test sur test (le découpage officiel de Phish360)
  rq2  entraînement sur la période "ancienne", test sur la période "recente" (tâche 3)

Sorties :
  results/baselines.csv        F1, précision, rappel, ROC-AUC, PR-AUC par modèle et protocole
  results/gbt_importance.csv   importance des features techniques (GBT, protocole rq1)
  results/preds_baselines.parquet   probabilité "phishing" de chaque page de test (pour la tâche 10)

Usage :
    python src/baselines.py                 # les 2 protocoles, 4 cœurs
    python src/baselines.py --cv            # + validation croisée 3 plis pour régler la régularisation
    python src/baselines.py --only rq1
"""
import argparse
import csv
import os
import time
from datetime import datetime

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import (IDF, HashingTF, NGram, RegexTokenizer, StandardScaler,
                                StopWordsRemover, VectorAssembler)
from pyspark.ml.functions import vector_to_array
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

FEAT = "data/features"
NON_FEATURES = {"sample_id", "split", "label", "chunk"}


def load(spark):
    """Une ligne par page : url, texte, 35 features techniques, label (1 = phishing), période RQ2."""
    url = spark.read.parquet(f"{FEAT}/url_feat").select("sample_id", "url")
    html = spark.read.parquet(f"{FEAT}/html_text").select(
        "sample_id", F.concat_ws(" ", F.coalesce("title", F.lit("")), F.coalesce("text", F.lit(""))).alias("doc"))
    tech = spark.read.parquet(f"{FEAT}/tech_feat").drop("chunk")
    tech_cols = [c for c in tech.columns if c not in NON_FEATURES]
    rq2 = spark.read.parquet("data/rq2_split.parquet").select("sample_id", "rq2_period")
    df = (tech.join(url, "sample_id").join(html, "sample_id").join(rq2, "sample_id", "left")
              .fillna(0, subset=tech_cols)
              .withColumn("y", (F.col("label") == "phishing").cast("double"))
              # l'URL coupée en caractères : "paypal.com" -> ["p","a","y",...]
              .withColumn("url_chars", F.expr("filter(split(lower(url), ''), c -> c != '')")))
    return df, tech_cols


def url_stages():
    return [NGram(n=3, inputCol="url_chars", outputCol="url_3g"),
            NGram(n=4, inputCol="url_chars", outputCol="url_4g"),
            HashingTF(inputCol="url_3g", outputCol="url_tf3", numFeatures=1 << 16),
            HashingTF(inputCol="url_4g", outputCol="url_tf4", numFeatures=1 << 16),
            IDF(inputCol="url_tf3", outputCol="url_idf3", minDocFreq=2),
            IDF(inputCol="url_tf4", outputCol="url_idf4", minDocFreq=2),
            VectorAssembler(inputCols=["url_idf3", "url_idf4"], outputCol="url_vec")]


def html_stages():
    stop = StopWordsRemover.loadDefaultStopWords("english") + StopWordsRemover.loadDefaultStopWords("french")
    return [RegexTokenizer(inputCol="doc", outputCol="words", pattern=r"[^\p{L}\p{N}<>]+", minTokenLength=2),
            StopWordsRemover(inputCol="words", outputCol="words_f", stopWords=stop),
            HashingTF(inputCol="words_f", outputCol="html_tf", numFeatures=1 << 17),
            IDF(inputCol="html_tf", outputCol="html_vec", minDocFreq=2)]


def tech_stages(tech_cols):
    return [VectorAssembler(inputCols=tech_cols, outputCol="tech_raw"),
            # mise à l'échelle pour la régression logistique (le GBT n'en a pas besoin)
            StandardScaler(inputCol="tech_raw", outputCol="tech_vec", withMean=False, withStd=True)]


def lr(features):
    return LogisticRegression(featuresCol=features, labelCol="y", maxIter=100, regParam=0.01,
                              elasticNetParam=0.0)


def models(tech_cols):
    return {
        "url_tfidf": (url_stages(), lr("url_vec")),
        "html_tfidf": (html_stages(), lr("html_vec")),
        "tech_gbt": ([VectorAssembler(inputCols=tech_cols, outputCol="tech_raw")],
                     GBTClassifier(featuresCol="tech_raw", labelCol="y", maxIter=100, maxDepth=5, seed=42)),
        "concat": (url_stages() + html_stages() + tech_stages(tech_cols)
                   + [VectorAssembler(inputCols=["url_vec", "html_vec", "tech_vec"], outputCol="all_vec")],
                   lr("all_vec")),
    }


def metrics(pred):
    """Phishing = classe positive. Les AUC sont calculées par Spark, le reste à partir de la matrice de confusion."""
    c = pred.select(
        F.sum(((F.col("prediction") == 1) & (F.col("y") == 1)).cast("int")).alias("tp"),
        F.sum(((F.col("prediction") == 1) & (F.col("y") == 0)).cast("int")).alias("fp"),
        F.sum(((F.col("prediction") == 0) & (F.col("y") == 1)).cast("int")).alias("fn"),
        F.count("*").alias("n")).first()
    prec = c.tp / (c.tp + c.fp) if c.tp + c.fp else 0.0
    rec = c.tp / (c.tp + c.fn) if c.tp + c.fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    ev = BinaryClassificationEvaluator(labelCol="y", rawPredictionCol="rawPrediction")
    return {"n_test": c.n, "f1": f1, "precision": prec, "recall": rec,
            "roc_auc": ev.setMetricName("areaUnderROC").evaluate(pred),
            "pr_auc": ev.setMetricName("areaUnderPR").evaluate(pred)}


def fit(stages, clf, train, cv, cores):
    if cv and isinstance(clf, LogisticRegression):
        grid = ParamGridBuilder().addGrid(clf.regParam, [0.001, 0.01, 0.1]).build()
        ev = BinaryClassificationEvaluator(labelCol="y", metricName="areaUnderPR")
        clf = CrossValidator(estimator=clf, estimatorParamMaps=grid, evaluator=ev, numFolds=3,
                             parallelism=1, seed=42)
    return Pipeline(stages=stages + [clf]).fit(train)


def log_time(step, pages, cores, secs):
    path = "results/timings.csv"
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "etape", "pages", "coeurs", "secondes", "pages_par_s"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), step, pages, cores,
                    round(secs, 2), round(pages / secs, 1) if secs else 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["rq1", "rq2"])
    ap.add_argument("--cv", action="store_true", help="validation croisée 3 plis (plus long)")
    ap.add_argument("--cores", default="4")
    ap.add_argument("--memory", default="4g")
    ap.add_argument("--pause", type=int, default=15, help="secondes de pause entre deux modèles")
    args = ap.parse_args()

    spark = (SparkSession.builder.master(f"local[{args.cores}]").appName("phish360_baselines")
             .config("spark.driver.memory", args.memory)
             .config("spark.sql.shuffle.partitions", str(int(args.cores) * 4))
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    cores = spark.sparkContext.defaultParallelism
    os.makedirs("results", exist_ok=True)

    df, tech_cols = load(spark)
    df = df.cache()
    protocols = {
        "rq1": (df.where("split = 'trainval'"), df.where("split = 'test'")),
        "rq2": (df.where("rq2_period = 'ancienne'"), df.where("rq2_period = 'recente'")),
    }
    rows, preds = [], []
    todo = [args.only] if args.only else list(protocols)
    for proto in todo:
        train, test = protocols[proto]
        n_train = train.count()
        print(f"\n=== {proto} : {n_train} pages d'entraînement, {test.count()} de test ===")
        for name, (stages, clf) in models(tech_cols).items():
            t0 = time.perf_counter()
            model = fit(stages, clf, train, args.cv, cores)
            secs = time.perf_counter() - t0
            pred = model.transform(test).cache()
            m = metrics(pred)
            rows.append({"protocole": proto, "modele": name, **m, "train_s": round(secs, 1)})
            print(f"{name:11s} F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
                  f"ROC-AUC={m['roc_auc']:.3f}  PR-AUC={m['pr_auc']:.3f}  ({secs:.0f} s)")
            log_time(f"baseline_{proto}_{name}", n_train, cores, secs)
            preds.append(pred.select("sample_id", F.lit(proto).alias("protocole"), F.lit(name).alias("modele"),
                                     "y", vector_to_array("probability")[1].alias("p_phishing")).toPandas())
            if proto == "rq1" and name == "tech_gbt":
                imp = model.stages[-1].featureImportances.toArray()
                with open("results/gbt_importance.csv", "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["feature", "importance"])
                    for col, v in sorted(zip(tech_cols, imp), key=lambda x: -x[1]):
                        w.writerow([col, round(float(v), 4)])
            pred.unpersist()
            if args.pause:
                time.sleep(args.pause)

    with open("results/baselines.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
    import pandas as pd
    pd.concat(preds).to_parquet("results/preds_baselines.parquet", index=False)

    print("\nRésumé (results/baselines.csv) :")
    print(pd.DataFrame(rows).drop(columns=["n_test"]).round(3).to_string(index=False))
    if os.path.exists("results/gbt_importance.csv"):
        print("\nTop 10 des features techniques (GBT) :")
        print(pd.read_csv("results/gbt_importance.csv").head(10).to_string(index=False))
    spark.stop()


if __name__ == "__main__":
    main()
