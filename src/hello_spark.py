"""Tâche 1 - premier test : Spark lit 10 dossiers du dataset Phish360.

Usage :  python src/hello_spark.py "C:/Users/<vous>/bigdata/Phish360"
"""
import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

ROOT = sys.argv[1] if len(sys.argv) > 1 else "bigdata/Phish360"
N_SAMPLES = 10

spark = (
    SparkSession.builder.master("local[*]")
    .appName("phish360_hello")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# 1. Choisir 10 dossiers du split "test" (chacun = une page web)
test_dir = os.path.join(ROOT, "test")
samples = sorted(os.listdir(test_dir))[:N_SAMPLES]
paths = [os.path.join(test_dir, s) for s in samples]
print(f"{len(os.listdir(test_dir))} dossiers dans test/, on en lit {len(paths)}")

# 2. Spark lit tous les fichiers (url.txt, index.html, screen_shoot, label.txt) en binaire
files = (
    spark.read.format("binaryFile")
    .option("recursiveFileLookup", "true")
    .load(paths)
    .withColumn("parts", F.split("path", "/"))
    .withColumn("sample_id", F.element_at("parts", -3))   # ex. L00001_legitimate
    .withColumn("modality", F.element_at("parts", -2))    # Label, RAW-HTML, SCREEN-SHOT, URL
)

# 3. Une ligne par page : label, url, taille du HTML, taille de l'image
def text_of(modality):
    # decode + suppression du BOM UTF-8 (\ufeff) et des espaces / retours ligne Windows
    return F.max(F.when(F.col("modality") == modality,
                        F.regexp_replace(F.decode("content", "UTF-8"), r"^[\s\x{FEFF}]+|[\s\x{FEFF}]+$", "")))

def size_of(modality):
    return F.max(F.when(F.col("modality") == modality, F.col("length")))

pages = files.groupBy("sample_id").agg(
    text_of("Label").alias("label"),
    text_of("URL").alias("url"),
    size_of("RAW-HTML").alias("html_bytes"),
    size_of("SCREEN-SHOT").alias("img_bytes"),
).orderBy("sample_id")

pages.show(truncate=60)
pages.groupBy("label").count().show()
spark.stop()
