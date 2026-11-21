
from pathlib import Path
import csv
import torch

from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "mvtec_anomaly_detection"
CHECKPOINT_ROOT = PROJECT_ROOT / "results" / "Patchcore" / "MVTecAD"
OUTPUT_FILE = PROJECT_ROOT / "results" / "baseline_metrics.csv"

CATEGORIES = ["metal_nut", "screw", "transistor"]

METRIC_NAMES = [
    "image_AUROC",
    "image_F1Score",
    "pixel_AUROC",
    "pixel_F1Score",
]

COLUMNS = ["category", *METRIC_NAMES, "status"]


# ============================================================
# OUTILS
# ============================================================

def get_checkpoint(category):

    possible_paths = [
        CHECKPOINT_ROOT / category / "latest" / "weights"
        / "lightning" / "model.ckpt",

        CHECKPOINT_ROOT / category / "v0" / "weights"
        / "lightning" / "model.ckpt",
    ]

    for path in possible_paths:
        if path.exists():
            return path

    return None


def to_float(value):

    if value is None:
        return ""

    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())

    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


def save_results(rows):

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=COLUMNS,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# EVALUATION D'UNE CATEGORIE SUR LES IMAGES ORIGINALES
# ============================================================

def evaluate_category(category):

    print("\n" + "=" * 60)
    print(f"Catégorie : {category}")
    print("=" * 60)

    checkpoint = get_checkpoint(category)

    if checkpoint is None:
        raise FileNotFoundError(
            f"Checkpoint introuvable pour {category}"
        )

    print(f"Checkpoint utilisé : {checkpoint}")

    datamodule = MVTecAD(
        root=str(DATASET_ROOT),
        category=category,
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=0,
    )

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
    )

    engine = Engine(
        accelerator="cpu",
        devices=1,
    )

    print("Évaluation sur les images originales...")

    results = engine.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=str(checkpoint),
    )

    if not results:
        raise RuntimeError(
            f"Aucune métrique retournée pour {category}"
        )

    metrics = results[0] if isinstance(results, list) else results

    row = {
        "category": category,
        "status": "success",
    }

    for metric in METRIC_NAMES:
        row[metric] = to_float(metrics.get(metric))

    print("\nRésultats :")

    for metric in METRIC_NAMES:
        print(f"{metric} : {row[metric]}")

    return row


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    all_rows = []

    print("\nÉVALUATION PATCHCORE SUR LES IMAGES ORIGINALES")
    print("Aucun entraînement ne sera effectué.")

    for category in CATEGORIES:

        try:
            row = evaluate_category(category)

        except Exception as error:

            print(f"\nERREUR pour {category} : {error}")

            row = {
                "category": category,
                "status": f"error: {error}",
            }

            for metric in METRIC_NAMES:
                row[metric] = ""

        all_rows.append(row)

        # Sauvegarde après chaque catégorie.
        save_results(all_rows)

    successful = sum(
        row["status"] == "success"
        for row in all_rows
    )

    print("\n" + "=" * 60)
    print("ÉVALUATION DES IMAGES ORIGINALES TERMINÉE")
    print(f"Catégories évaluées avec succès : {successful}/3")
    print(f"Résultats : {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
