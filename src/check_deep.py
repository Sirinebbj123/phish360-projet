"""Sprint 2 - vérifie que l'environnement deep learning est prêt (à lancer avant les tâches 6 à 12).

Contrôle : PyTorch + CUDA (GPU, mémoire), torchvision, transformers, xgboost, shap, scipy,
les entrées du Sprint 1 (data/features/*, data/rq2_split.parquet), un petit calcul sur le GPU,
et (option --download) le téléchargement des poids DistilBERT et ResNet18.

Usage :
    python src/check_deep.py
    python src/check_deep.py --download      # télécharge une fois les modèles pré-entraînés (~ 600 Mo)
"""
import argparse
import importlib
import os
import sys

OK, KO, AV = "[OK]", "[KO]", "[!!]"


def line(tag, title, msg):
    print(f"{tag} {title}: {msg}")
    return tag != KO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--model", default="distilbert-base-multilingual-cased")
    args = ap.parse_args()
    good = [line(OK, "Python", sys.version.split()[0])]

    try:
        import torch
        good.append(line(OK, "PyTorch", torch.__version__))
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            x = torch.randn(2048, 2048, device="cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                (x @ x).sum().item()
            good.append(line(OK, "GPU", f"{p.name}, {p.total_memory / 2**30:.1f} Go, CUDA {torch.version.cuda}, calcul fp16 OK"))
        else:
            line(AV, "GPU", "aucun GPU vu par PyTorch : réinstallez torch avec l'index CUDA (voir guide), "
                 "sinon tâches 7-8 très lentes (options --freeze)")
    except Exception as e:  # noqa: BLE001
        good.append(line(KO, "PyTorch", e))

    for pkg in ["torchvision", "transformers", "tokenizers", "xgboost", "shap", "sklearn", "scipy", "pyarrow", "matplotlib"]:
        try:
            good.append(line(OK, pkg, importlib.import_module(pkg).__version__))
        except Exception as e:  # noqa: BLE001
            good.append(line(KO, pkg, e))

    for path in ["data/features/url_feat", "data/features/html_text", "data/features/img_224",
                 "data/features/tech_feat", "data/rq2_split.parquet"]:
        good.append(line(OK if os.path.exists(path) else KO, path,
                         "présent" if os.path.exists(path) else "absent : refaire les tâches 3-4 (ou copier data/)"))
    for path in ["results/baselines.csv", "results/preds_baselines.parquet"]:
        line(OK if os.path.exists(path) else AV, path,
             "présent" if os.path.exists(path) else "absent : lancer src/baselines.py pour comparer aux baselines")

    if args.download:
        try:
            from transformers import AutoModel, AutoTokenizer
            AutoTokenizer.from_pretrained(args.model)
            AutoModel.from_pretrained(args.model)
            good.append(line(OK, "DistilBERT", f"{args.model} téléchargé (cache Hugging Face)"))
        except Exception as e:  # noqa: BLE001
            good.append(line(KO, "DistilBERT", e))
        try:
            from torchvision.models import ResNet18_Weights, resnet18
            resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            good.append(line(OK, "ResNet18", "poids ImageNet téléchargés"))
        except Exception as e:  # noqa: BLE001
            good.append(line(KO, "ResNet18", e))

    print("\nPrêt pour le Sprint 2 !" if all(good) else "\nCorrigez les lignes [KO] puis relancez.")


if __name__ == "__main__":
    main()
