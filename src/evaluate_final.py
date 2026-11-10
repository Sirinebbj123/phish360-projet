"""Tâche 10 - évaluation finale et comparaison complète des modèles (RQ1, RQ2).

Rassemble TOUS les modèles à partir de leurs prédictions sur le test (results/preds_*.parquet) :
  baselines MLlib (tâche 5) : url_tfidf, html_tfidf, tech_gbt, concat
  deep unimodal (6, 7, 8)   : url_charcnn, html_distilbert, img_resnet18
  classique Python (9)      : tech_xgb
  fusion (9)                : fusion_stacking, fusion_concat, fusion_gated

Pour chaque protocole :
  - les 5 métriques recalculées de la même façon pour tous (seuil 0,5), avec un intervalle de
    confiance à 95 % du F1 et de la PR-AUC (bootstrap, 1 000 tirages des pages de test) ;
  - un test de significativité entre la meilleure fusion et chaque autre modèle, sur les mêmes pages :
    différence de F1 par bootstrap apparié + test de McNemar (erreurs différentes ou non ?) ;
  - RQ1 : la meilleure fusion bat-elle la meilleure modalité seule ?
  - RQ2 : baisse de F1 entre le contrôle aléatoire (rq2ctrl) et le vrai découpage temporel (rq2).

Sorties :
  results/final_comparaison.csv        une ligne par (protocole, modèle), avec les IC
  results/final_significativite.csv    meilleure fusion contre chaque modèle
  results/final_rq2_derive.csv         F1 rq1 / rq2ctrl / rq2 et la perte due au temps
  results/rapport_sprint2.md           tableaux + conclusions, prêts à copier dans le rapport
  results/figures/final_f1_<proto>.png, final_pr_<proto>.png, final_rq2_derive.png,
  results/figures/final_ablation.png, final_cout.png

Usage :
    python src/evaluate_final.py
    python src/evaluate_final.py --boot 2000
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deep_common as dc  # noqa: E402

# nom technique -> (nom lisible, famille, modalité)
MODELS = {
    "url_tfidf": ("URL TF-IDF + LR", "Baseline MLlib", "URL"),
    "html_tfidf": ("HTML TF-IDF + LR", "Baseline MLlib", "HTML"),
    "tech_gbt": ("Techniques GBT", "Baseline MLlib", "Technique"),
    "concat": ("Concaténation + LR", "Baseline MLlib", "Fusion"),
    "url_charcnn": ("URL CharCNN", "Deep unimodal", "URL"),
    "html_distilbert": ("HTML DistilBERT", "Deep unimodal", "HTML"),
    "html_distilbert_gele": ("HTML DistilBERT figé", "Deep unimodal", "HTML"),
    "img_resnet18": ("Image ResNet18", "Deep unimodal", "Image"),
    "img_resnet18_gele": ("Image ResNet18 figé", "Deep unimodal", "Image"),
    "tech_xgb": ("Techniques XGBoost", "Classique Python", "Technique"),
    "fusion_stacking": ("Fusion stacking", "Fusion deep", "Fusion"),
    "fusion_concat": ("Fusion concat (MLP)", "Fusion deep", "Fusion"),
    "fusion_gated": ("Fusion gated (MLP)", "Fusion deep", "Fusion"),
}
# couleurs par famille (ordre catégoriel fixe, palette validée daltonisme)
COLORS = {"Fusion deep": "#2a78d6", "Deep unimodal": "#eb6834", "Baseline MLlib": "#1baf7a",
          "Classique Python": "#eda100"}
INK, MUTED = "#0b0b0b", "#52514e"
PROTO_TXT = {"rq1": "RQ1 : trainval → test", "rq2": "RQ2 : ancienne → récente",
             "rq2ctrl": "Contrôle RQ2 : même taille, tirage aléatoire"}


def style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="x", alpha=0.25)


def load_preds():
    frames = [pd.read_parquet(f) for f in sorted(glob.glob("results/preds_*.parquet"))]
    if not frames:
        raise SystemExit("Aucun fichier results/preds_*.parquet : lancez d'abord les tâches 5 à 9.")
    p = pd.concat(frames, ignore_index=True)
    p["y"] = p.y.astype(int)
    return p.drop_duplicates(["protocole", "modele", "sample_id"], keep="last")


def boot_ci(y, p, n, rng):
    """IC à 95 % du F1 et de la PR-AUC en retirant les pages de test au hasard (avec remise)."""
    pred = (p >= 0.5).astype(int)
    f1s, aps = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        f1s.append(f1_score(y[i], pred[i], zero_division=0))
        aps.append(average_precision_score(y[i], p[i]))
    q = lambda a: (np.percentile(a, 2.5), np.percentile(a, 97.5))  # noqa: E731
    return q(f1s), q(aps)


def paired(ya, pa, pb, n, rng):
    """Différence de F1 (A - B) par bootstrap apparié + McNemar exact sur les erreurs au seuil 0,5."""
    a, b = (pa >= 0.5).astype(int), (pb >= 0.5).astype(int)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(ya), len(ya))
        diffs.append(f1_score(ya[i], a[i], zero_division=0) - f1_score(ya[i], b[i], zero_division=0))
    ok_a, ok_b = a == ya, b == ya
    n01, n10 = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
    p_mc = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return (f1_score(ya, a) - f1_score(ya, b), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5),
            n01, n10, p_mc)


def extra_columns():
    """train_s, seuil_val, f1_seuil_val depuis baselines.csv et deep_results.csv."""
    out = []
    for f in ["results/baselines.csv", "results/deep_results.csv"]:
        if os.path.exists(f):
            out.append(pd.read_csv(f))
    if not out:
        return pd.DataFrame(columns=["protocole", "modele"])
    d = pd.concat(out, ignore_index=True).drop_duplicates(["protocole", "modele"], keep="last")
    keep = [c for c in ["protocole", "modele", "train_s", "seuil_val", "f1_seuil_val", "f1_graines_std",
                        "appareil"] if c in d]
    return d[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000, help="tirages bootstrap")
    args = ap.parse_args()
    os.makedirs("results/figures", exist_ok=True)
    rng = np.random.default_rng(42)
    preds = load_preds()
    preds = preds[preds.modele.isin(MODELS)]

    rows = []
    for (proto, model), g in preds.groupby(["protocole", "modele"]):
        y, p = g.y.to_numpy(), g.p_phishing.to_numpy()
        m = dc.metrics(y, p)
        (f1_lo, f1_hi), (ap_lo, ap_hi) = boot_ci(y, p, args.boot, rng)
        name, fam, mod = MODELS[model]
        rows.append({"protocole": proto, "modele": model, "nom": name, "famille": fam, "modalite": mod,
                     "n_test": m["n_test"], "f1": m["f1"], "f1_ic_bas": f1_lo, "f1_ic_haut": f1_hi,
                     "precision": m["precision"], "recall": m["recall"], "roc_auc": m["roc_auc"],
                     "pr_auc": m["pr_auc"], "pr_auc_ic_bas": ap_lo, "pr_auc_ic_haut": ap_hi})
    res = pd.DataFrame(rows).merge(extra_columns(), on=["protocole", "modele"], how="left")
    res = res.sort_values(["protocole", "f1"], ascending=[True, False])
    res.round(4).to_csv("results/final_comparaison.csv", index=False)

    md = ["# Sprint 2 : comparaison finale des modèles (généré par evaluate_final.py)\n",
          "F1, précision, rappel : classe phishing, seuil 0,5. IC 95 % par bootstrap des pages de test.\n"]
    sig_rows, verdicts = [], []
    for proto in [p for p in ["rq1", "rq2", "rq2ctrl"] if p in set(res.protocole)]:
        r = res[res.protocole == proto]
        md += [f"\n## {PROTO_TXT[proto]}\n",
               "| Modèle | Famille | F1 [IC 95 %] | Précision | Rappel | ROC-AUC | PR-AUC | F1 seuil val. | Entraînement (s) |",
               "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for _, x in r.iterrows():
            f1sv = f"{x.f1_seuil_val:.3f}" if pd.notna(x.get("f1_seuil_val")) else "-"
            ts = f"{x.train_s:.0f}" if pd.notna(x.get("train_s")) else "-"
            md.append(f"| {x.nom} | {x.famille} | {x.f1:.3f} [{x.f1_ic_bas:.3f}-{x.f1_ic_haut:.3f}] | "
                      f"{x.precision:.3f} | {x.recall:.3f} | {x.roc_auc:.3f} | {x.pr_auc:.3f} | {f1sv} | {ts} |")
        print(f"\n=== {PROTO_TXT[proto]} ===")
        print(r[["nom", "f1", "f1_ic_bas", "f1_ic_haut", "precision", "recall", "roc_auc", "pr_auc"]]
              .round(3).to_string(index=False))

        # significativité : meilleure fusion (famille Fusion deep) contre tous les autres
        fus = r[r.famille == "Fusion deep"]
        if len(fus):
            best = fus.iloc[0].modele
            pa = preds[(preds.protocole == proto) & (preds.modele == best)].set_index("sample_id")
            for other in r.modele:
                if other == best:
                    continue
                pb = preds[(preds.protocole == proto) & (preds.modele == other)].set_index("sample_id")
                common = pa.index.intersection(pb.index)
                d, lo, hi, n01, n10, pmc = paired(pa.loc[common, "y"].to_numpy(), pa.loc[common, "p_phishing"].to_numpy(),
                                                  pb.loc[common, "p_phishing"].to_numpy(), args.boot, rng)
                sig_rows.append({"protocole": proto, "fusion": best, "contre": other, "pages": len(common),
                                 "delta_f1": d, "ic_bas": lo, "ic_haut": hi, "seule_fusion_juste": n01,
                                 "seul_autre_juste": n10, "p_mcnemar": pmc,
                                 "significatif": bool(lo > 0 or hi < 0)})
            sig = pd.DataFrame([s for s in sig_rows if s["protocole"] == proto])
            md += ["", f"Meilleure fusion : **{MODELS[best][0]}**. Comparaison appariée (mêmes pages de test) :", "",
                   "| Contre | ΔF1 (fusion - autre) | IC 95 % | p McNemar | Significatif |", "| --- | --- | --- | --- | --- |"]
            for _, s in sig.iterrows():
                md.append(f"| {MODELS[s.contre][0]} | {s.delta_f1:+.3f} | [{s.ic_bas:+.3f} ; {s.ic_haut:+.3f}] | "
                          f"{s.p_mcnemar:.3g} | {'oui' if s.significatif else 'non'} |")

            uni = r[r.modalite != "Fusion"]
            if len(uni):
                bu = uni.iloc[0]
                bf = fus.iloc[0]
                s = sig[sig.contre == bu.modele].iloc[0]
                v = (f"{PROTO_TXT[proto]} : meilleure fusion {bf.nom} F1 = {bf.f1:.3f} contre meilleure modalité seule "
                     f"{bu.nom} F1 = {bu.f1:.3f} (ΔF1 = {s.delta_f1:+.3f}, IC [{s.ic_bas:+.3f} ; {s.ic_haut:+.3f}]) -> "
                     + ("la fusion fait significativement mieux." if s.ic_bas > 0 else
                        "la fusion fait significativement moins bien." if s.ic_haut < 0 else
                        "différence non significative."))
                verdicts.append(v)
                md += ["", f"**Verdict.** {v}"]
    if sig_rows:
        pd.DataFrame(sig_rows).round(4).to_csv("results/final_significativite.csv", index=False)

    # RQ2 : dérive temporelle
    piv = res.pivot_table(index="modele", columns="protocole", values="f1")
    if {"rq2", "rq2ctrl"} <= set(piv.columns):
        der = piv.dropna(subset=["rq2", "rq2ctrl"]).copy()
        der["perte_temps"] = der.rq2ctrl - der.rq2
        der = der.sort_values("perte_temps")
        der.insert(0, "nom", [MODELS[m][0] for m in der.index])
        der.round(4).to_csv("results/final_rq2_derive.csv")
        md += ["\n## RQ2 : perte due au temps (F1 contrôle aléatoire - F1 temporel)\n",
               "| Modèle | F1 RQ1 | F1 contrôle | F1 RQ2 | Perte due au temps |", "| --- | --- | --- | --- | --- |"]
        for m, x in der.iterrows():
            f1r1 = f"{x.rq1:.3f}" if "rq1" in der and pd.notna(x.get("rq1")) else "-"
            md.append(f"| {x.nom} | {f1r1} | {x.rq2ctrl:.3f} | {x.rq2:.3f} | {x.perte_temps:+.3f} |")
        md.append("\nLe modèle qui perd le moins est le plus robuste aux nouvelles campagnes.")
        fig, ax = plt.subplots(figsize=(8, 0.45 * len(der) + 1.5))
        yy = np.arange(len(der))
        ax.barh(yy + 0.2, der.rq2ctrl, 0.38, color="#86b6ef", label="contrôle (aléatoire)")
        ax.barh(yy - 0.2, der.rq2, 0.38, color="#2a78d6", label="RQ2 (temporel)")
        ax.set_yticks(yy, der.nom)
        ax.set_xlim(max(0, der[["rq2", "rq2ctrl"]].min().min() - 0.1), 1)
        ax.set_xlabel("F1 (phishing)")
        ax.set_title("RQ2 : effet du temps à taille égale", loc="left", color=INK)
        ax.legend(frameon=False, loc="lower right")
        style(ax)
        fig.tight_layout(); fig.savefig("results/figures/final_rq2_derive.png", dpi=150); plt.close(fig)

    # Figures F1 avec IC et courbes précision-rappel
    for proto in ["rq1", "rq2"]:
        r = res[res.protocole == proto].sort_values("f1")
        if r.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 0.42 * len(r) + 1.6))
        yy = np.arange(len(r))
        ax.barh(yy, r.f1, color=[COLORS[f] for f in r.famille], height=0.6)
        ax.errorbar(r.f1, yy, xerr=[r.f1 - r.f1_ic_bas, r.f1_ic_haut - r.f1], fmt="none", ecolor=INK, lw=1, capsize=2)
        for yi, v in zip(yy, r.f1):
            ax.text(v + 0.012, yi, f"{v:.3f}", va="center", fontsize=8, color=INK)
        ax.set_yticks(yy, r.nom)
        ax.set_xlim(max(0, r.f1_ic_bas.min() - 0.05), 1.04)
        ax.set_xlabel("F1 (phishing), barre = IC 95 %")
        ax.set_title(PROTO_TXT[proto], loc="left", color=INK)
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for f, c in COLORS.items() if f in set(r.famille)]
        ax.legend(handles, [f for f in COLORS if f in set(r.famille)], frameon=False, loc="lower right", fontsize=8)
        style(ax)
        fig.tight_layout(); fig.savefig(f"results/figures/final_f1_{proto}.png", dpi=150); plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        best_per_fam = res[res.protocole == proto].sort_values("pr_auc", ascending=False).groupby("famille").head(1)
        for _, x in best_per_fam.iterrows():
            g = preds[(preds.protocole == proto) & (preds.modele == x.modele)]
            pr, rc, _ = precision_recall_curve(g.y, g.p_phishing)
            ax.plot(rc, pr, lw=2, color=COLORS[x.famille], label=f"{x.nom} (PR-AUC {x.pr_auc:.3f})")
        ax.set_xlabel("rappel"); ax.set_ylabel("précision"); ax.set_ylim(0.4, 1.01)
        ax.set_title(f"Précision-rappel, meilleur modèle par famille\n{PROTO_TXT[proto]}", loc="left", color=INK, fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
        style(ax); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(f"results/figures/final_pr_{proto}.png", dpi=150); plt.close(fig)

    # Ablation des modalités (tâche 9)
    if os.path.exists("results/fusion_ablation.csv"):
        ab = pd.read_csv("results/fusion_ablation.csv")
        ab1 = ab[(ab.protocole == "rq1")].pivot_table(index="modalites", columns="strategie", values="f1")
        if len(ab1):
            ab1 = ab1.sort_values(ab1.columns[-1])
            fig, ax = plt.subplots(figsize=(8, 0.35 * len(ab1) + 1.5))
            yy = np.arange(len(ab1))
            for k, (col, c) in enumerate(zip(ab1.columns, ["#2a78d6", "#eb6834"])):
                ax.barh(yy + (k - 0.5) * 0.38, ab1[col], 0.36, color=c, label=col)
            ax.set_yticks(yy, ab1.index)
            ax.set_xlim(max(0, ab1.min().min() - 0.05), 1)
            ax.set_xlabel("F1 (phishing)")
            ax.set_title("RQ1 : quelles modalités combiner ? (ablation)", loc="left", color=INK)
            ax.legend(frameon=False, loc="lower right")
            style(ax)
            fig.tight_layout(); fig.savefig("results/figures/final_ablation.png", dpi=150); plt.close(fig)
            md += ["\n## Ablation des modalités (RQ1, F1)\n", ab1.round(3).to_markdown()
                   if hasattr(ab1, "to_markdown") and _has_tabulate() else ab1.round(3).to_string()]

    # Coût d'entraînement (lien avec RQ3)
    r1 = res[(res.protocole == "rq1") & res.train_s.notna()].sort_values("train_s") if "train_s" in res else pd.DataFrame()
    if len(r1):
        fig, ax = plt.subplots(figsize=(8, 0.42 * len(r1) + 1.5))
        yy = np.arange(len(r1))
        ax.barh(yy, r1.train_s, color=[COLORS[f] for f in r1.famille], height=0.6)
        for yi, v in zip(yy, r1.train_s):
            ax.text(v * 1.08, yi, f"{v:.0f} s", va="center", fontsize=8, color=INK)
        ax.set_xscale("log"); ax.set_yticks(yy, r1.nom)
        ax.set_xlabel("temps d'entraînement RQ1 (s, échelle log)")
        ax.set_title("Coût d'entraînement des modèles", loc="left", color=INK)
        style(ax)
        fig.tight_layout(); fig.savefig("results/figures/final_cout.png", dpi=150); plt.close(fig)

    if verdicts:
        md += ["\n## Conclusions automatiques\n"] + [f"- {v}" for v in verdicts]
    with open("results/rapport_sprint2.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n" + "\n".join(verdicts))
    print("\n-> results/final_comparaison.csv, final_significativite.csv, rapport_sprint2.md, figures/final_*.png")


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
