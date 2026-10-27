"""Tâche 3 (1/3) - extrait des métadonnées légères de chaque page, en distribué.

Lit data/phish360_raw.parquet (lourd : HTML + images) et écrit data/phish360_meta.parquet
(léger : quelques Mo) qui servira à l'analyse exploratoire et au choix de l'axe temporel (RQ2).

Usage :  python src/extract_meta.py            (4 cœurs par défaut, comme l'ingestion)
"""
import argparse
import io
import time

from PIL import Image
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

# Années de copyright : "© 2019", "&copy; 2015-2021", "Copyright 2020"... on garde la dernière année
COPYRIGHT_RE = r"(?i)(?:©|&copy;|&#169;|copyright)[^0-9<]{0,30}(?:(?:19|20)[0-9]{2}\s*[-–]\s*)?((?:19|20)[0-9]{2})"
# Toute année 2000-2029 présente dans le HTML (plus bruité : versions de scripts, dates d'articles...)
ANY_YEAR_RE = r"\b(20[0-2][0-9])\b"


@F.udf(T.StructType([T.StructField("w", T.IntegerType()), T.StructField("h", T.IntegerType())]))
def jpeg_size(data):
    """Largeur / hauteur d'une capture JPEG (67 pages seulement). PIL ne lit que l'en-tête."""
    if data is None:
        return None
    try:
        w, h = Image.open(io.BytesIO(data)).size
        return (w, h)
    except Exception:  # noqa: BLE001 - image corrompue : on note None
        return None


def png_dim(offset):
    """Entier sur 4 octets (big-endian) à la position offset de l'image, si c'est un PNG."""
    raw = F.expr(f"conv(hex(substring(image, {offset}, 4)), 16, 10)").cast("int")
    return F.when(F.col("image_format") == "png", raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/phish360_raw.parquet")
    ap.add_argument("--out", default="data/phish360_meta.parquet")
    ap.add_argument("--cores", default="4")
    ap.add_argument("--memory", default="4g")
    args = ap.parse_args()

    spark = (SparkSession.builder.master(f"local[{args.cores}]").appName("phish360_meta")
             .config("spark.driver.memory", args.memory).getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    t0 = time.perf_counter()

    df = spark.read.parquet(args.src)
    years_cr = F.transform(F.regexp_extract_all("html", F.lit(COPYRIGHT_RE), F.lit(1)), lambda x: x.cast("int"))
    years_any = F.transform(F.regexp_extract_all("html", F.lit(ANY_YEAR_RE), F.lit(1)), lambda x: x.cast("int"))
    host = F.lower(F.expr("parse_url(url, 'HOST')"))

    meta = df.select(
        "sample_id", "split", "label", "brand", "url", "html_bytes", "image_bytes", "image_format",
        F.regexp_extract("sample_id", r"^[LP](\d+)", 1).cast("int").alias("id_num"),
        F.length("url").alias("url_len"),
        F.lower(F.col("url")).startswith("https").alias("https"),
        host.alias("host"),
        F.element_at(F.split(host, r"\."), -1).alias("tld"),
        F.array_max(F.filter(years_cr, lambda y: y.between(1995, 2026))).alias("year_copyright"),
        F.array_max(F.filter(years_any, lambda y: y.between(2000, 2026))).alias("year_max"),
        # PNG : largeur et hauteur sont écrites aux octets 17-24 (en-tête IHDR). On les lit
        # directement en Spark SQL, sans envoyer les images (≈ 4 Go) à Python.
        png_dim(17).alias("png_w"), png_dim(21).alias("png_h"),
        # JPEG : seules ces images passent par Python (les autres envoient NULL)
        jpeg_size(F.when(F.col("image_format") == "jpeg", F.col("image"))).alias("jpg"),
    )
    meta = (meta.withColumn("img_w", F.coalesce("png_w", "jpg.w"))
                .withColumn("img_h", F.coalesce("png_h", "jpg.h"))
                .drop("png_w", "png_h", "jpg"))

    # Pas de coalesce(1) : il ferait tout passer par un seul cœur (et un seul tampon mémoire)
    meta.write.mode("overwrite").parquet(args.out)
    n = spark.read.parquet(args.out).count()
    print(f"{n} lignes écrites dans {args.out} en {time.perf_counter() - t0:.0f} s")
    spark.stop()


if __name__ == "__main__":
    main()
