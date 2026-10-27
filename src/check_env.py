"""Tâche 1 - vérifie que l'environnement du projet Phish360 est prêt.

Usage :  python src/check_env.py
"""
import importlib
import os
import platform
import shutil
import subprocess
import sys

OK, KO = "[OK]", "[KO]"


def check(title, fn):
    try:
        msg = fn()
        print(f"{OK} {title}: {msg}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"{KO} {title}: {e}")
        return False


def python_version():
    v = sys.version_info
    if not (v.major == 3 and 9 <= v.minor <= 11):
        raise RuntimeError(f"Python {v.major}.{v.minor} : utilisez 3.10 ou 3.11 avec PySpark 3.5")
    return platform.python_version()


def java_version():
    java = shutil.which("java")
    if java is None:
        raise RuntimeError("java introuvable : installez Java 17 et ajoutez-le au PATH")
    out = subprocess.run([java, "-version"], capture_output=True, text=True).stderr
    return out.splitlines()[0] + f" | JAVA_HOME={os.environ.get('JAVA_HOME', 'non défini')}"


def hadoop_home():
    if os.name != "nt":
        return "inutile hors Windows"
    home = os.environ.get("HADOOP_HOME")
    if not home or not os.path.exists(os.path.join(home, "bin", "winutils.exe")):
        raise RuntimeError("HADOOP_HOME/bin/winutils.exe manquant (nécessaire pour écrire du Parquet sous Windows)")
    return home


def package(name):
    return lambda: importlib.import_module(name).__version__


def spark_session():
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.master("local[*]").appName("check_env").getOrCreate()
    n = spark.range(1_000_000).selectExpr("sum(id) as s").first()["s"]
    cores = spark.sparkContext.defaultParallelism
    spark.stop()
    return f"Spark {spark.version}, {cores} cœurs, somme test = {n}"


def torch_device():
    import torch

    if torch.cuda.is_available():
        return f"torch {torch.__version__}, GPU = {torch.cuda.get_device_name(0)}"
    return f"torch {torch.__version__}, pas de GPU (CPU seulement : prévoir Colab/Kaggle pour T7-T8)"


if __name__ == "__main__":
    results = [
        check("Python", python_version),
        check("Java", java_version),
        check("Hadoop (Windows)", hadoop_home),
    ]
    for pkg in ["pyspark", "pyarrow", "pandas", "bs4", "tldextract", "PIL", "sklearn", "transformers"]:
        results.append(check(pkg, package(pkg)))
    results.append(check("PyTorch", torch_device))
    results.append(check("Session Spark", spark_session))
    print("\nEnvironnement prêt !" if all(results) else "\nCorrigez les lignes [KO] puis relancez.")
