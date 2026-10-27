"""Tâche 2 - contrôle qualité du Parquet Phish360.

Usage :  python src/quality_report.py            (lit data/phish360_raw.parquet)
Sortie : affichage console + results/quality_report.json
"""
import json
import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

PATH = sys.argv[1] if len(sys.argv) > 1 else "data/phish360_raw.parquet"
SMALL_HTML = 1024  # octets : en dessous, la page est probablement vide ou une erreur

spark = (SparkSession.builder.master("local[*]").appName("phish360_quality")
         .config("spark.driver.memory", "4g").getOrCreate())
spark.sparkContext.setLogLevel("WARN")
# on ne charge pas les colonnes lourdes (html, image) : les tailles suffisent ici
df = spark.read.parquet(PATH).drop("image", "html").cache()
report = {}

def show(title, frame, n=20):
    print(f"\n=== {title} ===")
    frame.show(n, truncate=80)
    return [r.asDict() for r in frame.limit(n).collect()]

# 1. Volume et équilibre des classes
report["pages_par_split_label"] = show(
    "Pages par split et label",
    df.groupBy("split", "label").count().orderBy("split", "label"))

# 2. Modalités manquantes
report["modalites_manquantes"] = show(
    "Modalités manquantes",
    df.select(F.explode("missing").alias("modalite")).groupBy("modalite").count())

# 2 bis. Fichiers illisibles (souvent bloqués par l'antivirus)
report["fichiers_illisibles"] = show(
    "Fichiers illisibles",
    df.select("split", "label", F.explode("read_errors").alias("erreur"))
      .groupBy("split", "label", "erreur").count().orderBy(F.desc("count")))

# 2 ter. Marques les plus imitées (pages de phishing)
report["marques_imitees"] = show(
    "Top 15 des marques imitées",
    df.filter("brand IS NOT NULL").groupBy("brand").count().orderBy(F.desc("count")), 15)

# 3. label.txt cohérent avec le nom du dossier ?
#    légitime : label.txt = "legitimate" ; phishing : label.txt = la marque (ex. "dhl")
expected = F.when(F.col("label") == "legitimate", F.lit("legitimate")).otherwise(F.col("brand"))
bad = df.filter(~F.col("label_txt").eqNullSafe(expected))
report["labels_incoherents"] = bad.count()
print(f"\nLabels incohérents (label.txt ≠ nom du dossier) : {report['labels_incoherents']}")
bad.select("split", "sample_id", "label_txt").show(10, truncate=False)

# 4. HTML vides / très petits, formats d'image
report["html_trop_petits"] = show(
    f"HTML < {SMALL_HTML} octets",
    df.filter(F.col("html_bytes") < SMALL_HTML).groupBy("split", "label").count())
report["formats_image"] = show(
    "Formats d'image", df.groupBy("image_format").count())

# 5. Statistiques de taille
report["tailles"] = show(
    "Tailles (octets)",
    df.groupBy("label").agg(
        F.expr("percentile_approx(html_bytes, 0.5)").alias("html_median"),
        F.max("html_bytes").alias("html_max"),
        F.expr("percentile_approx(image_bytes, 0.5)").alias("image_median"),
        F.expr("percentile_approx(length(url), 0.5)").alias("url_len_median")))

# 6. Doublons d'URL : dans un même split, et ENTRE trainval et test (fuite de données)
urls = df.select("split", F.lower(F.col("url")).alias("u"))
dup_within = urls.groupBy("split", "u").count().filter("count > 1")
report["urls_dupliquees_meme_split"] = dup_within.groupBy("split").agg(
    F.count("*").alias("urls"), F.sum("count").alias("pages")).toPandas().to_dict("records")
print("\nURL en double dans un même split :", report["urls_dupliquees_meme_split"])

tr = urls.filter("split = 'trainval'").select("u").distinct()
te = urls.filter("split = 'test'").select("u").distinct()
report["urls_communes_train_test"] = tr.join(te, "u").count()
print(f"URL présentes à la fois dans trainval et test : {report['urls_communes_train_test']}")

# 7. Même chose au niveau du domaine (plus strict)
hosts = df.select("split", F.lower(F.expr("parse_url(url, 'HOST')")).alias("host"))
th = hosts.filter("split = 'trainval'").select("host").distinct()
eh = hosts.filter("split = 'test'").select("host").distinct()
report["domaines_communs_train_test"] = th.join(eh, "host").count()
report["domaines_test"] = eh.count()
print(f"Domaines du test déjà vus dans trainval : {report['domaines_communs_train_test']}"
      f" / {report['domaines_test']}")

# 8. Dates des fichiers (piste pour RQ2)
report["dates_fichiers"] = show(
    "Plage de dates des fichiers HTML",
    df.groupBy("split", "label").agg(
        F.min("html_mtime").alias("html_min"), F.max("html_mtime").alias("html_max"),
        F.countDistinct(F.to_date("html_mtime")).alias("jours_distincts")))

os.makedirs("results", exist_ok=True)
with open("results/quality_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False, default=str)
print("\nRapport écrit dans results/quality_report.json")
spark.stop()
