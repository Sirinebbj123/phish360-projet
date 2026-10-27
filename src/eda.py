"""Tâche 3 (2/3) - analyse exploratoire sur les métadonnées (data/phish360_meta.parquet).

Le fichier ne fait que quelques Mo : pandas suffit ici. Le travail lourd (lire 4 Go de HTML et
d'images) a déjà été fait en distribué par extract_meta.py.

Usage :  python src/eda.py
Sorties : results/figures/*.png  +  results/eda_summary.txt
"""
import os

import matplotlib
matplotlib.use("Agg")  # pas de fenêtre : on enregistre les images
import matplotlib.pyplot as plt
import pandas as pd

META = "data/phish360_meta.parquet"
FIG = "results/figures"
os.makedirs(FIG, exist_ok=True)
df = pd.read_parquet(META)
lines = []


def out(title, obj):
    text = f"\n=== {title} ===\n{obj if isinstance(obj, str) else obj.to_string()}"
    print(text)
    lines.append(text)


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, name), dpi=120)
    plt.close()


# 1. Équilibre des classes
counts = df.groupby(["split", "label"]).size().unstack()
out("Pages par split et label", counts)
counts.plot.bar(title="Pages par split et label")
save("01_classes.png")

# 2. Tailles et longueur d'URL : distributions par label (échelle log)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col, log in zip(axes, ["html_bytes", "image_bytes", "url_len"], [True, True, False]):
    for lab, g in df.groupby("label"):
        ax.hist(g[col].dropna(), bins=60, alpha=0.5, label=lab, log=False)
    if log:
        ax.set_xscale("log")
    ax.set_title(col)
    ax.legend()
save("02_tailles.png")
out("Médianes par label", df.groupby("label")[["html_bytes", "image_bytes", "url_len"]].median())

# 3. Dimensions des captures : un raccourci possible pour le CNN ?
dims = (df.assign(dim=df.img_w.astype("Int64").astype(str) + "x" + df.img_h.astype("Int64").astype(str))
          .groupby(["label", "dim"]).size().rename("pages").reset_index()
          .sort_values("pages", ascending=False).groupby("label").head(5))
out("Top 5 des dimensions de capture par label", dims)

# 4. HTTPS et domaines de premier niveau
out("Part d'URL en HTTPS par label", df.groupby("label")["https"].mean().round(3))
tld = (df.groupby("label")["tld"].value_counts(normalize=True).rename("part")
         .groupby(level=0).head(8).round(3))
out("Top 8 des TLD par label", tld)

# 5. Années trouvées dans le HTML : candidates pour l'axe temporel de RQ2
cov = df.groupby(["split", "label"])[["year_copyright", "year_max"]].apply(lambda g: g.notna().mean()).round(3)
out("Part des pages avec une année trouvée", cov)
out("Année de copyright : répartition par label (trainval + test)",
    pd.crosstab(df.year_copyright, df.label).tail(15))
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, sp in zip(axes, ["trainval", "test"]):
    pd.crosstab(df[df.split == sp].year_copyright, df[df.split == sp].label).tail(12).plot.bar(ax=ax)
    ax.set_title(f"Année de copyright - {sp}")
save("03_annees_copyright.png")

# 6. Les numéros de dossier suivent-ils le temps ? (année médiane par tranche de 1000 numéros)
df["id_bin"] = (df.id_num // 1000) * 1000
trend = df.groupby(["label", "id_bin"])["year_copyright"].agg(["median", "count"])
out("Année de copyright médiane par tranche de numéros", trend)
for lab, g in df.groupby("label"):
    m = g.groupby("id_bin")["year_copyright"].median()
    plt.plot(m.index, m.values, marker="o", label=lab)
plt.xlabel("numéro de dossier (tranches de 1000)")
plt.ylabel("année de copyright médiane")
plt.legend()
plt.title("Les numéros suivent-ils le temps ?")
save("04_numero_vs_annee.png")

# 7. Marques imitées (trainval seulement : le test est anonymisé en 'phish')
out("Top 15 des marques imitées (trainval)",
    df[(df.split == "trainval") & df.brand.notna()].brand.value_counts().head(15))

with open("results/eda_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\nGraphiques dans {FIG}/ et résumé dans results/eda_summary.txt")
