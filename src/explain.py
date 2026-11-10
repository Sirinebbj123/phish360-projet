"""Tâche 12 - explicabilité et analyse des erreurs.

Quatre éclairages, du plus global au plus fin :
  1. SHAP sur les features techniques (XGBoost de la tâche 9) : quelles features poussent vers
     "phishing" ou "légitime", et dans quel sens (graphique en essaim + importance moyenne).
  2. Poids des modalités dans la fusion : coefficients du stacking et "portes" moyennes du
     modèle gated (sur quelle modalité la fusion s'appuie-t-elle ?).
  3. Analyse des erreurs du meilleur modèle de fusion : faux positifs / faux négatifs, taux d'erreur
     par marque imitée, par quantité de texte, image manquante, et quelle modalité seule avait
     raison quand la fusion s'est trompée. Planche des captures des pires erreurs.
  4. (option) Explications des modèles deep :
       --gradcam N  Grad-CAM du ResNet18 : zones de la capture qui ont fait dire "phishing"
       --tokens N   attribution gradient x entrée de DistilBERT : mots qui ont le plus pesé

Sorties (results/ et results/figures/) :
  shap_tech_<proto>.png, shap_tech_importance_<proto>.csv, fusion_poids.png,
  erreurs_<proto>.csv, erreurs_resume_<proto>.txt, erreurs_marques_<proto>.png,
  erreurs_captures_<proto>.png, gradcam_<proto>.png, tokens_distilbert_<proto>.csv

Usage :
    python src/explain.py                         # rq1 et rq2, sans les options deep
    python src/explain.py --gradcam 6 --tokens 10 # avec Grad-CAM et mots importants (GPU conseillé)
"""
import argparse
import io
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deep_common as dc  # noqa: E402

INK, MUTED, BLUE, ORANGE = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"
NON_FEATURES = {"sample_id", "split", "label", "chunk"}


def style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)


# ---------- 1. SHAP ---------------------------------------------------------------

def shap_tech(df, proto, max_pages):
    from xgboost import XGBClassifier
    path = f"models/tech_xgb_{proto}.json"
    if not os.path.exists(path):
        print(f"  [!] {path} absent : lancez d'abord src/fusion.py")
        return
    clf = XGBClassifier()
    clf.load_model(path)
    tech = pd.read_parquet(f"{dc.FEAT}/tech_feat").drop_duplicates("sample_id")
    cols = [c for c in tech.columns if c not in NON_FEATURES]
    part = dc.protocol_parts(df, proto)
    test_ids = df.sample_id[part == "test"]
    X = tech[tech.sample_id.isin(test_ids)][cols].astype(float).fillna(0.0)
    if len(X) > max_pages:
        X = X.sample(max_pages, random_state=42)
    # valeurs SHAP exactes calculées par XGBoost lui-même (TreeSHAP) : évite les soucis de version entre shap et xgboost
    import xgboost as xgb
    sv = clf.get_booster().predict(xgb.DMatrix(X.to_numpy(np.float32), feature_names=list(cols)), pred_contribs=True)[:, :-1]
    imp = pd.DataFrame({"feature": cols, "shap_moyen_abs": np.abs(sv).mean(0),
                        "sens": [np.corrcoef(X[c], sv[:, j])[0, 1] if X[c].std() > 0 else 0
                                 for j, c in enumerate(cols)]})
    imp["sens"] = np.where(imp.sens > 0, "valeur haute -> phishing", "valeur haute -> légitime")
    imp = imp.sort_values("shap_moyen_abs", ascending=False)
    imp.round(4).to_csv(f"results/shap_tech_importance_{proto}.csv", index=False)
    try:
        import shap
    except ImportError:
        print("  [!] shap non installé : seul le CSV d'importance est produit")
        return
    plt.figure()
    shap.summary_plot(sv, X, max_display=15, show=False, plot_size=(8, 6))
    plt.title(f"SHAP, features techniques ({proto}) : à droite = pousse vers phishing", loc="left", fontsize=10)
    plt.tight_layout(); plt.savefig(f"results/figures/shap_tech_{proto}.png", dpi=150); plt.close("all")
    print("  SHAP top 8 :", ", ".join(imp.feature.head(8)))


# ---------- 2. poids des modalités ----------------------------------------------------

def plot_weights():
    if not os.path.exists("results/fusion_poids.csv"):
        return
    w = pd.read_csv("results/fusion_poids.csv")
    w = w[w.protocole.isin(["rq1", "rq2"])]
    strategies = list(dict.fromkeys(w.strategie))
    fig, axes = plt.subplots(1, len(strategies), figsize=(4 * len(strategies), 3.4), squeeze=False)
    for ax, s in zip(axes[0], strategies):
        d = w[w.strategie == s].pivot_table(index="modalite", columns="protocole", values="poids")
        xx = np.arange(len(d))
        for k, (col, c) in enumerate(zip(d.columns, [BLUE, ORANGE])):
            ax.bar(xx + (k - 0.5) * 0.38, d[col], 0.36, color=c, label=col)
        ax.set_xticks(xx, d.index); ax.set_title(s, fontsize=9, loc="left"); style(ax)
        ax.axhline(0, color=MUTED, lw=0.8)
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle("Poids des modalités dans la fusion", x=0.01, ha="left", color=INK)
    fig.tight_layout(); fig.savefig("results/figures/fusion_poids.png", dpi=150); plt.close(fig)


# ---------- 3. erreurs ----------------------------------------------------------------

def best_fusion(proto):
    if os.path.exists("results/final_comparaison.csv"):
        r = pd.read_csv("results/final_comparaison.csv")
        r = r[(r.protocole == proto) & (r.famille == "Fusion deep")]
        if len(r):
            return r.sort_values("f1", ascending=False).modele.iloc[0]
    return "fusion_stacking"


def errors(df, proto, n_show):
    model = best_fusion(proto)
    path = f"results/preds_{model}.parquet"
    if not os.path.exists(path):
        print(f"  [!] {path} absent")
        return None
    p = pd.read_parquet(path).query("protocole == @proto")[["sample_id", "y", "p_phishing"]]
    url = pd.read_parquet(f"{dc.FEAT}/url_feat", columns=["sample_id", "url", "suffix"]).drop_duplicates("sample_id")
    html = pd.read_parquet(f"{dc.FEAT}/html_text", columns=["sample_id", "title", "text_len"]).drop_duplicates("sample_id")
    d = p.merge(url, on="sample_id", how="left").merge(html, on="sample_id", how="left")
    for m in ["url", "html", "img"]:
        f = f"{dc.EMB}/{m}_{proto}.parquet"
        if os.path.exists(f):
            cols = ["sample_id", "p_phishing"] + (["img_ok"] if m == "img" else [])
            e = pd.read_parquet(f, columns=cols).rename(columns={"p_phishing": f"p_{m}"})
            d = d.merge(e, on="sample_id", how="left")
    g = f"results/fusion_portes_{proto}.parquet"
    if os.path.exists(g):
        d = d.merge(pd.read_parquet(g), on="sample_id", how="left")
    d["marque"] = np.where(d.sample_id.str.startswith("P"), d.sample_id.str.partition("_")[2].str.lower(), "(légitime)")
    d["pred"] = (d.p_phishing >= 0.5).astype(int)
    d["erreur"] = np.select([(d.pred == 1) & (d.y == 0), (d.pred == 0) & (d.y == 1)], ["faux positif", "faux négatif"], "")
    err = d[d.erreur != ""].copy()
    err["confiance"] = np.abs(err.p_phishing - 0.5) * 2
    err = err.sort_values(["erreur", "confiance"], ascending=[True, False])
    err.drop(columns=["pred"]).round(3).to_csv(f"results/erreurs_{proto}.csv", index=False)

    lines = [f"Analyse des erreurs : {model}, protocole {proto}",
             f"{len(d)} pages de test, {len(err)} erreurs : {(err.erreur == 'faux positif').sum()} faux positifs "
             f"(légitime classée phishing), {(err.erreur == 'faux négatif').sum()} faux négatifs (phishing manqué)", ""]
    bins = pd.cut(d.text_len.fillna(0), [-1, 0, 200, 1000, 1e9], labels=["aucun texte", "< 200 car.", "200-1000", "> 1000"])
    lines.append("Taux d'erreur selon la quantité de texte visible :")
    for b, x in d.groupby(bins, observed=True):
        lines.append(f"  {b:12s} {len(x):5d} pages  erreur {(x.erreur != '').mean():.1%}")
    if "img_ok" in d:
        lines.append("Taux d'erreur selon l'image :")
        for k, x in d.groupby(d.img_ok.fillna(0).astype(int)):
            lines.append(f"  {'image OK' if k else 'image absente':12s} {len(x):5d} pages  erreur {(x.erreur != '').mean():.1%}")
    ph = d[d.y == 1]
    by_brand = ph.groupby("marque").agg(pages=("y", "size"), manques=("erreur", lambda s: (s != "").sum()))
    by_brand["taux_manque"] = by_brand.manques / by_brand.pages
    top = by_brand[by_brand.pages >= 10].sort_values("taux_manque", ascending=False).head(12)
    lines += ["", "Marques imitées les plus souvent manquées (au moins 10 pages de test) :"]
    lines += [f"  {b:20s} {int(x.manques):3d}/{int(x.pages):3d}  ({x.taux_manque:.0%})" for b, x in top.iterrows()]
    present = [m for m in ["url", "html", "img"] if f"p_{m}" in err]
    if present:
        lines += ["", "Quand la fusion se trompe, quelle modalité seule avait raison ?"]
        for m in present:
            right = ((err[f"p_{m}"] >= 0.5).astype(int) == err.y).mean()
            lines.append(f"  {m:5s} juste sur {right:.0%} des erreurs de la fusion")
    if any(c.startswith("porte_") for c in d):
        lines += ["", "Porte moyenne (modèle gated) : bonnes réponses contre erreurs"]
        for c in [c for c in d if c.startswith("porte_")]:
            lines.append(f"  {c[6:]:5s} juste {d.loc[d.erreur == '', c].mean():.2f}   erreur {d.loc[d.erreur != '', c].mean():.2f}")
    lines += ["", "Faux négatifs les plus sûrs d'eux (le modèle était certain que c'était légitime) :"]
    for _, x in err[err.erreur == "faux négatif"].head(8).iterrows():
        lines.append(f"  p={x.p_phishing:.2f}  {x.marque:12s} {str(x.url)[:70]}  | {str(x.title)[:40]}")
    lines += ["", "Faux positifs les plus sûrs d'eux :"]
    for _, x in err[err.erreur == "faux positif"].head(8).iterrows():
        lines.append(f"  p={x.p_phishing:.2f}  {str(x.url)[:70]}  | {str(x.title)[:40]}")
    txt = "\n".join(lines)
    with open(f"results/erreurs_resume_{proto}.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)

    if len(top):
        fig, ax = plt.subplots(figsize=(7, 0.35 * len(top) + 1.4))
        t = top.iloc[::-1]
        ax.barh(t.index, t.taux_manque, color=BLUE, height=0.6)
        for yi, (v, k, n) in enumerate(zip(t.taux_manque, t.manques, t.pages)):
            ax.text(v + 0.01, yi, f"{int(k)}/{int(n)}", va="center", fontsize=8, color=INK)
        ax.set_xlabel("part des pages de phishing manquées")
        ax.set_title(f"Marques imitées les plus manquées ({proto})", loc="left", color=INK)
        style(ax)
        fig.tight_layout(); fig.savefig(f"results/figures/erreurs_marques_{proto}.png", dpi=150); plt.close(fig)

    montage(err, proto, n_show)
    return err


def load_imgs(ids):
    img = pd.read_parquet(f"{dc.FEAT}/img_224", columns=["sample_id", "img", "img_ok"])
    img = img[img.sample_id.isin(ids)].drop_duplicates("sample_id").set_index("sample_id")
    return {i: (Image.open(io.BytesIO(img.loc[i, "img"])).convert("RGB") if i in img.index and img.loc[i, "img_ok"] == 1
                else Image.new("RGB", (224, 224), (230, 230, 230))) for i in ids}


def montage(err, proto, n):
    rows = [("faux négatif", "Phishing manqué"), ("faux positif", "Légitime bloquée")]
    picks = {k: err[err.erreur == k].head(n) for k, _ in rows}
    if not sum(len(v) for v in picks.values()):
        return
    ims = load_imgs(pd.concat(picks.values()).sample_id.tolist())
    fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 5), squeeze=False)
    for r, (k, lab) in enumerate(rows):
        for c in range(n):
            ax = axes[r][c]
            ax.axis("off")
            if c < len(picks[k]):
                x = picks[k].iloc[c]
                ax.imshow(ims[x.sample_id])
                ax.set_title(f"{x.marque[:12]}\np={x.p_phishing:.2f}", fontsize=7)
        axes[r][0].text(-0.1, 0.5, lab, transform=axes[r][0].transAxes, rotation=90, va="center", ha="right", fontsize=9)
    fig.suptitle(f"Erreurs les plus sûres d'elles ({proto})", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(f"results/figures/erreurs_captures_{proto}.png", dpi=130); plt.close(fig)


# ---------- 4. deep : Grad-CAM et mots importants ---------------------------------------

def gradcam(df, proto, n, device):
    import torch
    from img_resnet18 import MEAN, STD, ImgClassifier, transforms
    path = f"models/img_resnet18_{proto}.pt"
    if not os.path.exists(path):
        print(f"  [!] {path} absent")
        return
    model = ImgClassifier(pretrained=False).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    preds = pd.read_parquet("results/preds_img_resnet18.parquet").query("protocole == @proto and y == 1")
    picks = pd.concat([preds.sort_values("p_phishing", ascending=False).head(n),      # phishing bien reconnus
                       preds.sort_values("p_phishing").head(n)])                         # phishing manqués
    ims = load_imgs(picks.sample_id.tolist())
    layer = model.backbone.layer4
    store = {}
    h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__("act", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("grad", go[0]))
    tf = transforms(False)
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 5), squeeze=False)
    for k, (_, x) in enumerate(picks.iterrows()):
        im = ims[x.sample_id]
        t = tf(im).unsqueeze(0).to(device).requires_grad_(True)
        model.zero_grad()
        model(t).sum().backward()
        w = store["grad"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * store["act"]).sum(1)).squeeze().detach().cpu().numpy()
        cam = np.array(Image.fromarray((cam / (cam.max() + 1e-8) * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)) / 255
        ax = axes[k // n][k % n]
        ax.imshow(im); ax.imshow(cam, cmap="inferno", alpha=0.45); ax.axis("off")
        ax.set_title(f"{x.sample_id.partition('_')[2][:12]} p={x.p_phishing:.2f}", fontsize=7)
    h1.remove(); h2.remove()
    axes[0][0].text(-0.1, 0.5, "phishing reconnu", transform=axes[0][0].transAxes, rotation=90, va="center", ha="right")
    axes[1][0].text(-0.1, 0.5, "phishing manqué", transform=axes[1][0].transAxes, rotation=90, va="center", ha="right")
    fig.suptitle(f"Grad-CAM ResNet18 ({proto}) : zones claires = ce qui a fait monter le score phishing", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(f"results/figures/gradcam_{proto}.png", dpi=130); plt.close(fig)
    print(f"  Grad-CAM -> results/figures/gradcam_{proto}.png")


def tokens(df, proto, n, device):
    import torch
    from transformers import AutoTokenizer
    from html_distilbert import HtmlClassifier, load_texts
    path = f"models/html_distilbert_{proto}.pt"
    if not os.path.exists(path):
        print(f"  [!] {path} absent")
        return
    ck = torch.load(path, map_location=device, weights_only=False)
    tok = AutoTokenizer.from_pretrained(ck["model_name"])
    model = HtmlClassifier(ck["model_name"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    preds = pd.read_parquet("results/preds_html_distilbert.parquet").query("protocole == @proto")
    picks = pd.concat([preds[preds.y == 1].sort_values("p_phishing", ascending=False).head(n),
                       preds[(preds.y == 1) & (preds.p_phishing < 0.5)].head(n),
                       preds[(preds.y == 0) & (preds.p_phishing >= 0.5)].head(n)])
    sub = df[df.sample_id.isin(picks.sample_id)].reset_index(drop=True)
    docs = dict(zip(sub.sample_id, load_texts(sub)))
    out = []
    for _, x in picks.iterrows():
        enc = tok(docs[x.sample_id], truncation=True, max_length=ck["max_len"], return_tensors="pt").to(device)
        emb = model.bert.embeddings.word_embeddings(enc["input_ids"]).detach().requires_grad_(True)
        h = model.bert(inputs_embeds=emb, attention_mask=enc["attention_mask"]).last_hidden_state[:, 0]
        model.head(model.proj(h)).sum().backward()
        score = (emb.grad * emb).sum(-1).squeeze(0).detach().cpu().numpy()     # > 0 : pousse vers phishing
        words = tok.convert_ids_to_tokens(enc["input_ids"][0])
        best = [(w, s) for w, s in sorted(zip(words, score), key=lambda t: -t[1]) if w not in tok.all_special_tokens][:10]
        cas = "phishing reconnu" if x.y == 1 and x.p_phishing >= 0.5 else "phishing manqué" if x.y == 1 else "faux positif"
        out.append({"sample_id": x.sample_id, "cas": cas, "p_phishing": round(float(x.p_phishing), 3),
                    "mots_vers_phishing": " ".join(w for w, _ in best)})
    pd.DataFrame(out).to_csv(f"results/tokens_distilbert_{proto}.csv", index=False)
    print(pd.DataFrame(out).to_string(index=False, max_colwidth=70))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocoles", nargs="+", default=["rq1", "rq2"], choices=dc.PROTOCOLS)
    ap.add_argument("--shap-pages", type=int, default=2000, help="pages de test utilisées pour SHAP")
    ap.add_argument("--captures", type=int, default=8, help="captures par ligne dans la planche d'erreurs")
    ap.add_argument("--gradcam", type=int, default=0, help="nb de captures expliquées par Grad-CAM (0 = non)")
    ap.add_argument("--tokens", type=int, default=0, help="nb de pages par cas pour les mots DistilBERT (0 = non)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    dc.setup_dirs()
    df = dc.load_index()
    plot_weights()
    for proto in args.protocoles:
        print(f"\n=== {proto} ===")
        shap_tech(df, proto, args.shap_pages)
        errors(df, proto, args.captures)
        if args.gradcam or args.tokens:
            device = dc.device_of(args.cpu)
            if args.gradcam:
                gradcam(df, proto, args.gradcam, device)
            if args.tokens:
                tokens(df, proto, args.tokens, device)
    print("\n-> results/erreurs_*.csv, erreurs_resume_*.txt, shap_tech_importance_*.csv, figures/")


if __name__ == "__main__":
    main()
