# Détection de phishing multimodale (Phish360) : Big Data + Deep Learning

Projet de cours : détecter les pages de phishing en combinant 4 modalités (URL, HTML,
capture d'écran, features techniques) avec un pipeline distribué Spark et des modèles deep.

## Questions de recherche

- **RQ1** : la fusion des modalités fait-elle mieux que chaque modalité seule ?
- **RQ2** : la fusion généralise-t-elle mieux dans le temps, sur des campagnes plus récentes ?
- **RQ3** : quel est le coût de calcul du pipeline distribué quand le volume augmente ?

## Données

Dataset **Phish360** (trainval : 7 977 pages, test : 2 771). Les données ne sont **pas** dans ce dépôt
(volume et pages de phishing réelles). Placez-les dans `C:\bigdata\Phish360` (ou un autre dossier).

## Installation (Windows, VS Code)

Prérequis : Python 3.11, Java 17 (Temurin), `HADOOP_HOME` avec `winutils.exe` et `hadoop.dll`.

```powershell
.\setup_venv.ps1 -Gpu        # crée .venv, installe requirements.txt et torch (CUDA)
.\.venv\Scripts\Activate.ps1
python src\check_env.py      # tout doit afficher [OK]
```

## Pipeline

| Tâche | Script | Rôle | Sortie |
| --- | --- | --- | --- |
| 1 | `check_env.py`, `hello_spark.py` | vérifier l'environnement | - |
| 2 | `ingest.py`, `quality_report.py` | ingestion distribuée vers Parquet, par lots | `data/phish360_raw.parquet` |
| 3 | `extract_meta.py`, `eda.py`, `clean_index.py`, `rq2_split.py` | EDA, nettoyage, axe temporel RQ2 | `data/clean_index.parquet`, `data/rq2_split.parquet` |
| 4 | `preprocess.py` | prétraitement distribué des 4 modalités | `data/features/{url_feat,html_text,img_224,tech_feat}` |
| 5 | `baselines.py` | baselines Spark MLlib (TF-IDF, GBT, concaténation) | `results/baselines.csv` |

```powershell
python src\ingest.py "C:/bigdata/Phish360"
python src\quality_report.py
python src\extract_meta.py
python src\eda.py
python src\clean_index.py
python src\rq2_split.py
python src\preprocess.py
python src\baselines.py
```

Les scripts Spark tournent par défaut sur 4 cœurs, par lots, avec des pauses (portable qui chauffe) ;
un lot terminé n'est pas refait si on relance. Les temps sont enregistrés dans `results/timings.csv` (RQ3).

## Premiers résultats (baselines MLlib, F1 sur la classe phishing)

| Modèle | RQ1 (trainval → test) | RQ2 (ancienne → récente) |
| --- | --- | --- |
| URL TF-IDF + LR | 0,807 | 0,694 |
| HTML TF-IDF + LR | 0,913 | 0,884 |
| Features techniques + GBT | 0,934 | 0,867 |
| Concaténation + LR | 0,934 | 0,846 |
