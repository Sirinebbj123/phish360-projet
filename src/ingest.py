"""Tâche 2 - ingestion distribuée de Phish360 vers Parquet.

Chaque dossier (une page web) devient UNE ligne : id, split, label, url, html, image, tailles, dates.
Spark répartit les dossiers entre les cœurs ; chaque cœur lit ses fichiers en parallèle.

Usage :
    python src/ingest.py "C:/bigdata/Phish360"                 # tout le dataset (4 cœurs, lots de 500 pages)
    python src/ingest.py "C:/bigdata/Phish360" --limit 200     # test rapide sur 200 pages
    python src/ingest.py "C:/bigdata/Phish360" --cores 8       # plus rapide si le PC ne chauffe pas

Le travail est découpé en lots (dossiers chunk=000, chunk=001, ...). Si le PC s'arrête,
relancez la même commande : les lots déjà terminés sont sautés.
"""
import argparse
import csv
import os
import time
from datetime import datetime, timezone

from pyspark.sql import SparkSession
from pyspark.sql import types as T

SPLITS = ["trainval", "test"]
MODALITIES = {"label": "Label", "url": "URL", "html": "RAW-HTML", "image": "SCREEN-SHOT"}

SCHEMA = T.StructType([
    T.StructField("sample_id", T.StringType()),       # ex. L00001_legitimate
    T.StructField("split", T.StringType()),           # trainval / test
    T.StructField("brand", T.StringType()),           # marque imitée (ex. P19936_dhl -> dhl)
    T.StructField("label", T.StringType()),           # legitimate / phishing (préfixe L / P du dossier)
    T.StructField("label_txt", T.StringType()),       # contenu brut de label.txt ("legitimate" ou la marque)
    T.StructField("url", T.StringType()),
    T.StructField("html", T.StringType()),
    T.StructField("image", T.BinaryType()),
    T.StructField("image_format", T.StringType()),    # png / jpeg / autre
    T.StructField("html_bytes", T.LongType()),
    T.StructField("image_bytes", T.LongType()),
    T.StructField("url_mtime", T.TimestampType()),    # dates des fichiers : piste pour RQ2
    T.StructField("html_mtime", T.TimestampType()),
    T.StructField("image_mtime", T.TimestampType()),
    T.StructField("missing", T.ArrayType(T.StringType())),  # modalités absentes
    T.StructField("read_errors", T.ArrayType(T.StringType())),  # fichiers illisibles (antivirus...)
])


def first_file(folder):
    """Premier fichier d'un sous-dossier (le nom exact de la capture varie : screen_shoot.png, .jpg...)."""
    if not os.path.isdir(folder):
        return None
    names = sorted(n for n in os.listdir(folder) if os.path.isfile(os.path.join(folder, n)))
    return os.path.join(folder, names[0]) if names else None


def read_text(path):
    with open(path, "rb") as f:
        raw = f.read()
    # utf-8-sig retire le BOM ; errors="replace" évite qu'un octet invalide fasse planter le job
    return raw.decode("utf-8-sig", errors="replace").strip(), len(raw)


def mtime(path):
    return datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc).replace(tzinfo=None)


def image_format(head):
    if head.startswith(b"\x89PNG"):
        return "png"
    if head.startswith(b"\xff\xd8"):
        return "jpeg"
    return "autre"


def safe(fn, path, kind, errors):
    """Lit un fichier sans faire planter tout le job : l'erreur est notée dans read_errors.
    Sous Windows, Defender bloque souvent les pages de phishing (OSError Errno 22)."""
    try:
        return fn(path)
    except OSError as e:
        errors.append(f"{kind}: {type(e).__name__} {e.errno}")
        return None


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def read_sample(item):
    """Exécuté sur les cœurs Spark : lit les 4 fichiers d'un dossier et renvoie une ligne."""
    split, sample_dir = item
    sample_id = os.path.basename(sample_dir)
    files = {k: first_file(os.path.join(sample_dir, v)) for k, v in MODALITIES.items()}
    missing = [k for k, p in files.items() if p is None]
    errors = []

    label_txt = url = html = image = None
    html_bytes = image_bytes = None
    url_mt = html_mt = image_mt = None
    fmt = None
    if files["label"]:
        r = safe(read_text, files["label"], "label", errors)
        label_txt = r[0].lower() if r else None
    if files["url"]:
        r = safe(read_text, files["url"], "url", errors)
        url = r[0] if r else None
        url_mt = safe(mtime, files["url"], "url_mtime", errors)
    if files["html"]:
        r = safe(read_text, files["html"], "html", errors)
        if r:
            html, html_bytes = r
        html_mt = safe(mtime, files["html"], "html_mtime", errors)
    if files["image"]:
        image = safe(read_bytes, files["image"], "image", errors)
        if image is not None:
            image_bytes = len(image)
            fmt = image_format(image[:8])
        image_mt = safe(mtime, files["image"], "image_mtime", errors)

    # Nom du dossier : L00001_legitimate ou P19936_dhl (P = phishing, suivi de la marque imitée)
    prefix, _, suffix = sample_id.partition("_")
    label = {"L": "legitimate", "P": "phishing"}.get(prefix[:1].upper())
    brand = suffix.lower() if label == "phishing" else None
    return (sample_id, split, brand, label, label_txt, url, html, image, fmt,
            html_bytes, image_bytes, url_mt, html_mt, image_mt, missing, errors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="dossier Phish360 (contient trainval/ et test/)")
    ap.add_argument("--out", default="data/phish360_raw.parquet")
    ap.add_argument("--cores", default="4", help="nombre de cœurs Spark (* = tous)")
    ap.add_argument("--memory", default="4g", help="mémoire Spark (ex. 4g, 6g, 8g)")
    ap.add_argument("--chunk", type=int, default=500, help="pages par lot")
    ap.add_argument("--pause", type=int, default=20, help="secondes de pause entre deux lots (refroidissement)")
    ap.add_argument("--limit", type=int, default=0, help="nb max de pages par split (0 = toutes)")
    args = ap.parse_args()

    spark = (
        SparkSession.builder.master(f"local[{args.cores}]")
        .appName("phish360_ingest")
        .config("spark.driver.memory", args.memory)
        .config("spark.sql.parquet.compression.codec", "snappy")
        # groupes de lignes Parquet de 32 Mo au lieu de 128 : chaque cœur garde moins d'images en mémoire
        .config("spark.hadoop.parquet.block.size", str(32 * 1024 * 1024))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    t0 = time.perf_counter()

    # 1. Lister les dossiers (rapide : on ne lit que les noms)
    items = []
    for split in SPLITS:
        split_dir = os.path.join(args.root, split)
        names = sorted(os.listdir(split_dir))
        if args.limit and args.limit < len(names):
            names = names[: args.limit // 2] + names[-(args.limit - args.limit // 2):]  # L... et P...
        items += [(split, os.path.join(split_dir, n)) for n in names
                  if os.path.isdir(os.path.join(split_dir, n))]
    print(f"{len(items)} dossiers à ingérer")

    # 2. Découper en lots. Chaque lot est distribué sur les cœurs puis écrit dans son propre
    #    dossier chunk=XXX ; un lot terminé contient un fichier _SUCCESS et n'est pas refait.
    #    Spark relit ensuite tous les lots d'un coup (colonne "chunk" ajoutée automatiquement).
    chunks = [items[i:i + args.chunk] for i in range(0, len(items), args.chunk)]
    for k, chunk in enumerate(chunks):
        out_k = os.path.join(args.out, f"chunk={k:03d}")
        if os.path.exists(os.path.join(out_k, "_SUCCESS")):
            print(f"lot {k + 1}/{len(chunks)} déjà fait, on passe")
            continue
        t_k = time.perf_counter()
        n_parts = max(8, spark.sparkContext.defaultParallelism * 4)
        rdd = spark.sparkContext.parallelize(chunk, n_parts).map(read_sample)
        # 3. Écrire en Parquet. Pas de partitionBy : il obligerait Spark à trier chaque tâche
        #    (images comprises) en mémoire avant l'écriture -> OutOfMemoryError.
        spark.createDataFrame(rdd, SCHEMA).write.mode("overwrite").parquet(out_k)
        print(f"lot {k + 1}/{len(chunks)} : {len(chunk)} pages en {time.perf_counter() - t_k:.0f} s")
        if k < len(chunks) - 1 and args.pause:
            time.sleep(args.pause)
    elapsed = time.perf_counter() - t0

    n = spark.read.parquet(args.out).count()
    cores = spark.sparkContext.defaultParallelism
    print(f"{n} pages dans {args.out} ; cette exécution : {elapsed:.1f} s avec {cores} cœurs")

    # 4. Journaliser le temps pour RQ3 (tâche 11)
    os.makedirs("results", exist_ok=True)
    log = "results/timings.csv"
    new = not os.path.exists(log)
    with open(log, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "etape", "pages", "coeurs", "secondes", "pages_par_s"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), "ingestion", n, cores,
                    round(elapsed, 2), round(n / elapsed, 1)])
    # NB : ce temps inclut les pauses ; les mesures propres pour RQ3 se feront en tâche 11.
    spark.stop()


if __name__ == "__main__":
    main()
