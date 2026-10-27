"""Tâche 4 - prétraitement distribué des 4 modalités.

Lit data/phish360_raw.parquet (uniquement les pages gardées par data/clean_index.parquet)
et écrit un Parquet par modalité dans data/features/ :

  url_feat   URL découpée (schéma, sous-domaine, domaine, suffixe, chemin, paramètres) + mesures lexicales
  html_text  titre + texte visible (pour le modèle texte) + comptage des balises sensibles
  img_224    capture redimensionnée en 224 x 224, stockée en JPEG (≈ 15 Ko au lieu de 600 Ko)
  tech_feat  ~30 features numériques (URL + HTML + tailles) pour les modèles classiques (tâche 5)

La normalisation des images (moyenne / écart-type ImageNet) se fera au chargement en tâche 8 :
stocker des tenseurs float prendrait 224*224*3*4 = 600 Ko par page, soit ≈ 6 Go.

Comme l'ingestion, le travail suit les lots chunk=XXX du Parquet brut : un lot terminé
(fichier _SUCCESS) n'est pas refait, et une pause laisse refroidir le PC entre deux lots.

Usage :
    python src/preprocess.py                  # 4 cœurs, pause de 20 s
    python src/preprocess.py --only html      # une seule modalité (url, html, image)
"""
import argparse
import csv
import io
import ipaddress
import math
import os
import re
import time
from collections import Counter
from datetime import datetime
from urllib.parse import urlsplit, parse_qsl

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

SUSPICIOUS = ["login", "signin", "logon", "verify", "account", "update", "secure", "bank",
              "confirm", "password", "webscr", "billing", "wallet", "support", "unlock"]
TEXT_MAX = 4000          # caractères de texte visible gardés (un transformer en lit ~512 tokens)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
HTML_MAX = 2_000_000     # au-delà, on ne parse que les 2 premiers Mo (pages énormes = lenteur)

_extract = None


def tld_extract(host):
    """tldextract avec la liste de suffixes embarquée (aucun téléchargement depuis les cœurs Spark)."""
    global _extract
    if _extract is None:
        import tldextract
        _extract = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)
    return _extract(host)


def entropy(s):
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def is_ip(host):
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def same_site(link_host, page_domain):
    return link_host == "" or link_host == page_domain or link_host.endswith("." + page_domain)


def registered(host):
    ext = tld_extract(host)
    return ".".join(p for p in (ext.domain, ext.suffix) if p) or host


# ---------- URL ----------------------------------------------------------------

URL_SCHEMA = T.StructType([
    T.StructField("sample_id", T.StringType()), T.StructField("split", T.StringType()),
    T.StructField("label", T.StringType()), T.StructField("url", T.StringType()),
    T.StructField("scheme", T.StringType()), T.StructField("subdomain", T.StringType()),
    T.StructField("domain", T.StringType()), T.StructField("suffix", T.StringType()),
    T.StructField("path", T.StringType()), T.StructField("query", T.StringType()),
    *[T.StructField(c, T.IntegerType()) for c in [
        "url_len", "host_len", "path_len", "n_dots", "n_hyphens", "n_digits", "n_special",
        "n_subdomains", "n_params", "n_slashes", "has_ip", "has_at", "has_port", "https",
        "n_susp_words"]],
    T.StructField("url_entropy", T.DoubleType()),
])


def url_row(sample_id, split, label, url):
    url = (url or "").strip()
    full = url if "://" in url else "http://" + url       # urlsplit a besoin d'un schéma
    try:
        p = urlsplit(full)
        host, port = (p.hostname or ""), p.port
    except ValueError:                                    # URL mal formée (port invalide...)
        p, host, port = urlsplit("http://invalid"), "", None
    ext = tld_extract(host)
    low = url.lower()
    return {
        "sample_id": sample_id, "split": split, "label": label, "url": url,
        "scheme": p.scheme, "subdomain": ext.subdomain, "domain": ext.domain, "suffix": ext.suffix,
        "path": p.path, "query": p.query,
        "url_len": len(url), "host_len": len(host), "path_len": len(p.path),
        "n_dots": url.count("."), "n_hyphens": url.count("-"),
        "n_digits": sum(ch.isdigit() for ch in url),
        "n_special": len(re.findall(r"[^A-Za-z0-9./:\-]", url)),
        "n_subdomains": len([s for s in ext.subdomain.split(".") if s]),
        "n_params": len(parse_qsl(p.query)), "n_slashes": url.count("/"),
        "has_ip": int(is_ip(host)), "has_at": int("@" in url), "has_port": int(port is not None),
        "https": int(low.startswith("https")),
        "n_susp_words": sum(w in low for w in SUSPICIOUS),
        "url_entropy": entropy(host),
    }


def url_map(r):
    d = url_row(r.sample_id, r.split, r.label, r.url)
    return tuple(d[c] for c in URL_SCHEMA.names)


# ---------- HTML ---------------------------------------------------------------

HTML_INT = ["n_forms", "n_inputs", "n_password", "n_hidden", "n_scripts", "n_ext_scripts",
            "n_iframes", "n_links", "n_ext_links", "n_null_links", "n_imgs", "form_ext_action",
            "form_empty_action", "has_meta_refresh", "has_title", "text_len", "parse_error"]
HTML_SCHEMA = T.StructType([
    T.StructField("sample_id", T.StringType()), T.StructField("split", T.StringType()),
    T.StructField("label", T.StringType()), T.StructField("title", T.StringType()),
    T.StructField("text", T.StringType()),
    *[T.StructField(c, T.IntegerType()) for c in HTML_INT],
    T.StructField("ext_link_ratio", T.DoubleType()),
])


def link_host(href):
    try:
        return (urlsplit(href).hostname or "").lower()
    except ValueError:
        return ""


def html_row(sample_id, split, label, url, html):
    from bs4 import BeautifulSoup

    row = {"sample_id": sample_id, "split": split, "label": label, "title": None, "text": None,
           **{c: 0 for c in HTML_INT}, "ext_link_ratio": 0.0}
    try:
        page_dom = registered(urlsplit(url if "://" in (url or "") else "http://" + (url or "")).hostname or "")
        soup = BeautifulSoup((html or "")[:HTML_MAX], "lxml")
        forms = soup.find_all("form")
        inputs = soup.find_all("input")
        scripts = soup.find_all("script")
        n_ext_scripts = sum(bool(sc.get("src")) for sc in scripts)   # avant decompose() ci-dessous
        links = [a.get("href", "").strip() for a in soup.find_all("a")]
        abs_links = [h for h in links if h.lower().startswith(("http://", "https://", "//"))]
        ext = [h for h in abs_links if not same_site(link_host(h if "://" in h else "http:" + h), page_dom)]
        actions = [(f.get("action") or "").strip() for f in forms]
        title = soup.title.get_text(" ", strip=True) if soup.title else ""

        for tag in soup(["script", "style", "noscript", "template"]):
            tag.decompose()
        text = re.sub(r"\s+", " ", soup.get_text(" ")).strip()
        # les années (copyright...) définissent les périodes de RQ2 : on les masque pour que
        # le modèle texte ne puisse pas "tricher" en lisant l'année au lieu du contenu
        text, title = YEAR_RE.sub("<annee>", text), YEAR_RE.sub("<annee>", title)

        row.update({
            "title": title[:300], "text": text[:TEXT_MAX],
            "n_forms": len(forms), "n_inputs": len(inputs),
            "n_password": sum((i.get("type") or "").lower() == "password" for i in inputs),
            "n_hidden": sum((i.get("type") or "").lower() == "hidden" for i in inputs),
            "n_scripts": len(scripts), "n_ext_scripts": n_ext_scripts,
            "n_iframes": len(soup.find_all("iframe")), "n_links": len(links),
            "n_ext_links": len(ext),
            "n_null_links": sum(h in ("", "#") or h.lower().startswith("javascript") for h in links),
            "n_imgs": len(soup.find_all("img")),
            # formulaire qui envoie les données vers un autre domaine : signe classique de phishing
            "form_ext_action": int(any(a.lower().startswith(("http", "//"))
                                       and not same_site(link_host(a if "://" in a else "http:" + a), page_dom)
                                       for a in actions)),
            "form_empty_action": int(any(a in ("", "#", "about:blank") for a in actions)),
            "has_meta_refresh": int(soup.find("meta", attrs={"http-equiv": re.compile("refresh", re.I)}) is not None),
            "has_title": int(bool(title)), "text_len": len(text),
            "ext_link_ratio": len(ext) / len(links) if links else 0.0,
        })
    except Exception:  # noqa: BLE001 - HTML illisible : la ligne reste avec des zéros
        row["parse_error"] = 1
    return row


def html_map(r):
    d = html_row(r.sample_id, r.split, r.label, r.url, r.html)
    return tuple(d[c] for c in HTML_SCHEMA.names)


# ---------- Image --------------------------------------------------------------

IMG_SCHEMA = T.StructType([
    T.StructField("sample_id", T.StringType()), T.StructField("split", T.StringType()),
    T.StructField("label", T.StringType()), T.StructField("img", T.BinaryType()),
    T.StructField("img_ok", T.IntegerType()),
])


def img_224(data):
    from PIL import Image

    im = Image.open(io.BytesIO(data)).convert("RGB")       # RGBA / palette -> RGB
    im = im.resize((224, 224), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def img_map(r):
    try:
        return (r.sample_id, r.split, r.label, img_224(r.image), 1)
    except Exception:  # noqa: BLE001 - image corrompue
        return (r.sample_id, r.split, r.label, None, 0)


# ---------- Pipeline -----------------------------------------------------------

STEPS = {  # modalité -> (colonnes lues dans le brut, fonction, schéma)
    "url": (["url"], url_map, URL_SCHEMA),
    "html": (["url", "html"], html_map, HTML_SCHEMA),
    "image": (["image"], img_map, IMG_SCHEMA),
}
OUT_NAME = {"url": "url_feat", "html": "html_text", "image": "img_224"}


def log_time(step, pages, cores, secs):
    os.makedirs("results", exist_ok=True)
    path = "results/timings.csv"
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "etape", "pages", "coeurs", "secondes", "pages_par_s"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), step, pages, cores,
                    round(secs, 2), round(pages / secs, 1) if secs else 0])


def build_tech(spark, feat_dir):
    """~30 features numériques : URL + structure HTML + tailles des fichiers."""
    url = spark.read.parquet(os.path.join(feat_dir, "url_feat")).drop("chunk")
    html = spark.read.parquet(os.path.join(feat_dir, "html_text")).drop("chunk", "title", "text", "split", "label")
    sizes = spark.read.parquet("data/phish360_meta.parquet").select("sample_id", "html_bytes", "image_bytes")
    num_url = [f.name for f in URL_SCHEMA if isinstance(f.dataType, (T.IntegerType, T.DoubleType))]
    num_html = HTML_INT[:-1] + ["ext_link_ratio"]          # parse_error n'est pas une feature
    tech = (url.select("sample_id", "split", "label", *num_url)
               .join(html.select("sample_id", *num_html), "sample_id")
               .join(sizes, "sample_id", "left")
               .withColumn("html_kb", F.round(F.col("html_bytes") / 1024, 1))
               .withColumn("image_kb", F.round(F.col("image_bytes") / 1024, 1))
               .drop("html_bytes", "image_bytes"))
    out = os.path.join(feat_dir, "tech_feat")
    tech.write.mode("overwrite").parquet(out)
    return spark.read.parquet(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/phish360_raw.parquet")
    ap.add_argument("--index", default="data/clean_index.parquet")
    ap.add_argument("--out", default="data/features")
    ap.add_argument("--only", choices=list(STEPS), help="une seule modalité")
    ap.add_argument("--cores", default="4")
    ap.add_argument("--memory", default="4g")
    ap.add_argument("--pause", type=int, default=20, help="secondes de pause entre deux lots")
    args = ap.parse_args()

    spark = (SparkSession.builder.master(f"local[{args.cores}]").appName("phish360_preprocess")
             .config("spark.driver.memory", args.memory)
             .config("spark.hadoop.parquet.block.size", str(32 * 1024 * 1024))
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    cores = spark.sparkContext.defaultParallelism

    keep = (spark.read.parquet(args.index).where("keep").select("sample_id", "split"))
    chunks = sorted(d for d in os.listdir(args.src) if d.startswith("chunk="))
    steps = [args.only] if args.only else list(STEPS)
    print(f"{len(chunks)} lots x {len(steps)} modalités ({', '.join(steps)}) sur {cores} cœurs")

    for step in steps:
        cols, fn, schema = STEPS[step]
        t_step, n_step = time.perf_counter(), 0
        for k, chunk in enumerate(chunks):
            out_k = os.path.join(args.out, OUT_NAME[step], chunk)
            if os.path.exists(os.path.join(out_k, "_SUCCESS")):
                print(f"[{step}] {chunk} déjà fait, on passe")
                continue
            t_k = time.perf_counter()
            # jointure "broadcast" : l'index (petit) est copié sur chaque cœur, aucun tri des HTML/images
            raw = spark.read.parquet(os.path.join(args.src, chunk)).select("sample_id", "split", "label", *cols)
            df = raw.join(F.broadcast(keep), ["sample_id", "split"])
            # pas de repartition : chaque fichier du lot est déjà une tâche, et un repartition
            # ferait voyager les HTML et les images entre les cœurs (shuffle = mémoire + disque)
            # chaque cœur reçoit ses lignes en Python (comme pour l'ingestion) et renvoie une ligne par page
            spark.createDataFrame(df.rdd.map(fn), schema).write.mode("overwrite").parquet(out_k)
            n = spark.read.parquet(out_k).count()
            n_step += n
            print(f"[{step}] {chunk} : {n} pages en {time.perf_counter() - t_k:.0f} s")
            if args.pause and k < len(chunks) - 1:
                time.sleep(args.pause)
        secs = time.perf_counter() - t_step
        if n_step:
            log_time(f"pretraitement_{step}", n_step, cores, secs)   # inclut les pauses (cf. tâche 11)

    if os.path.isdir(os.path.join(args.out, "url_feat")) and os.path.isdir(os.path.join(args.out, "html_text")):
        tech = build_tech(spark, args.out)
        print(f"\ntech_feat : {tech.count()} pages x {len(tech.columns) - 3} features")

    print("\nRésumé :")
    for name in OUT_NAME.values():
        path = os.path.join(args.out, name)
        if os.path.isdir(path):
            print(f"  {name:10s} {spark.read.parquet(path).count():6d} pages")
    spark.stop()


if __name__ == "__main__":
    main()
