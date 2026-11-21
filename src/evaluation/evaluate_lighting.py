
from pathlib import Path
import csv
import torch

from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "mvtec_anomaly_detection"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_ROOT = PROJECT_ROOT / "results" / "Patchcore" / "MVTecAD"

OUTPUT_FILE = PROJECT_ROOT / "results" / "lighting_evaluation.csv"

CATEGORIES = ["metal_nut", "screw", "transistor"]

SHIFTS = [
    "brightness",
    "contrast",
    "gamma",
    "color_temperature",
    "shading",
]

SEVERITIES = [1, 2, 3]

METRIC_NAMES = [
    "image_AUROC",
    "image_F1Score",
    "pixel_AUROC",
    "pixel_F1Score",
]

COLUMNS = [
    "category",
    "shift",
    "severity",
    *METRIC_NAMES,
    "status",
]


# ============================================================
# OUTILS
# ============================================================

def get_checkpoint(category):
    return (
        CHECKPOINT_ROOT
        / category
        / "latest"
        / "weights"
        / "lightning"
        / "model.ckpt"
    )


def to_float(value):
    if value is None:
        return ""

    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())

    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


def get_result_key(row):
    """Identifie une configuration par catégorie, shift et sévérité."""
    return (
        row["category"],
        row["shift"],
        int(float(row["severity"])),
    )


def load_existing_results():
    """Charge les résultats déjà présents dans le CSV."""

    if not OUTPUT_FILE.exists():
        return {}

    results = {}

    with OUTPUT_FILE.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            try:
                key = get_result_key(row)

                # Ignorer les configurations qui ne font pas partie
                # de notre protocole expérimental.
                if (
                    key[0] not in CATEGORIES
                    or key[1] not in SHIFTS
                    or key[2] not in SEVERITIES
                ):
                    continue

                normalized_row = {
                    column: row.get(column, "")
                    for column in COLUMNS
                }

                normalized_row["severity"] = key[2]

                results[key] = normalized_row

            except (KeyError, TypeError, ValueError):
                print("Attention : une ligne invalide a été ignorée.")

    return results


def sort_results(results):
    """Organise les résultats dans l'ordre des catégories et transformations."""

    category_order = {
        category: index
        for index, category in enumerate(CATEGORIES)
    }

    shift_order = {
        shift: index
        for index, shift in enumerate(SHIFTS)
    }

    return sorted(
        results.values(),
        key=lambda row: (
            category_order[row["category"]],
            shift_order[row["shift"]],
            int(float(row["severity"])),
        ),
    )


def save_results(results):
    """Sauvegarde tous les résultats après chaque configuration."""

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    rows = sort_results(results)

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
# EVALUATION D'UNE CONFIGURATION
# ============================================================

def evaluate_configuration(category, shift, severity, checkpoint):

    print("\n" + "=" * 60)
    print(f"Catégorie : {category}")
    print(f"Variation : {shift}")
    print(f"Sévérité  : {severity}")
    print("=" * 60)

    transformed_root = (
        PROCESSED_ROOT
        / category
        / "test"
        / shift
        / f"severity_{severity}"
    )

    masks_root = (
        PROCESSED_ROOT
        / category
        / "masks"
        / shift
        / f"severity_{severity}"
    )

    normal_train = DATASET_ROOT / category / "train" / "good"
    normal_test = transformed_root / "good"

    if not normal_train.exists():
        raise FileNotFoundError(
            f"Dossier normal introuvable : {normal_train}"
        )

    if not normal_test.exists():
        raise FileNotFoundError(
            f"Dossier de test introuvable : {normal_test}"
        )

    # Récupérer les dossiers de défauts dans un ordre stable.
    defect_dirs = sorted(
        path
        for path in transformed_root.iterdir()
        if path.is_dir() and path.name != "good"
    )

    if not defect_dirs:
        raise FileNotFoundError(
            f"Aucun dossier de défaut trouvé dans {transformed_root}"
        )

    # Les dossiers de masques suivent exactement le même ordre.
    mask_dirs = [
        masks_root / defect_dir.name
        for defect_dir in defect_dirs
    ]

    for mask_dir in mask_dirs:
        if not mask_dir.exists():
            raise FileNotFoundError(
                f"Dossier de masques introuvable : {mask_dir}"
            )

    # Création du datamodule.
    datamodule = Folder(
        name=f"{category}_{shift}_{severity}",
        normal_dir=str(normal_train),
        normal_test_dir=str(normal_test),
        abnormal_dir=[str(path) for path in defect_dirs],
        mask_dir=[str(path) for path in mask_dirs],
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=0,
        test_split_mode="from_dir",
        val_split_mode="none",
    )

    # Même architecture et mêmes paramètres que le modèle entraîné.
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

    print("Évaluation en cours...")

    results = engine.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=str(checkpoint),
    )

    if not results:
        raise RuntimeError(
            "Anomalib n'a retourné aucune métrique."
        )

    metrics = results[0] if isinstance(results, list) else results

    row = {
        "category": category,
        "shift": shift,
        "severity": severity,
        "status": "success",
    }

    for metric_name in METRIC_NAMES:
        row[metric_name] = to_float(
            metrics.get(metric_name)
        )

    print("\nRésultats :")

    for metric_name in METRIC_NAMES:
        print(f"{metric_name} : {row[metric_name]}")

    return row


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    # Charger les résultats déjà enregistrés.
    results_by_key = load_existing_results()

    total = (
        len(CATEGORIES)
        * len(SHIFTS)
        * len(SEVERITIES)
    )

    successful_before = sum(
        row.get("status") == "success"
        for row in results_by_key.values()
    )

    print("\nREPRISE DE L'ÉVALUATION PATCHCORE")
    print("=" * 60)
    print(f"Configurations déjà enregistrées : {len(results_by_key)}/{total}")
    print(f"Configurations déjà réussies : {successful_before}")
    print(f"Configurations restantes à vérifier : {total - successful_before}")
    print("=" * 60)

    # Parcourir les catégories.
    for category in CATEGORIES:

        checkpoint = get_checkpoint(category)

        if not checkpoint.exists():
            print(f"\nCheckpoint absent pour {category}.")
            print(f"Chemin recherché : {checkpoint}")
            print("Cette catégorie sera ignorée.")
            continue

        print(f"\nCheckpoint utilisé : {checkpoint}")

        # Parcourir les transformations et les sévérités.
        for shift in SHIFTS:
            for severity in SEVERITIES:

                key = (category, shift, severity)

                # Ne pas recalculer les configurations réussies.
                previous_row = results_by_key.get(key)

                if (
                    previous_row is not None
                    and previous_row.get("status") == "success"
                ):
                    print(
                        f"\nDéjà évalué : {category} / "
                        f"{shift} / severity {severity}"
                    )
                    continue

                try:
                    row = evaluate_configuration(
                        category,
                        shift,
                        severity,
                        checkpoint,
                    )

                except Exception as error:

                    print(f"\nERREUR : {error}")

                    row = {
                        "category": category,
                        "shift": shift,
                        "severity": severity,
                        "status": f"error: {error}",
                    }

                    for metric_name in METRIC_NAMES:
                        row[metric_name] = ""

                # Ajouter ou remplacer le résultat de cette configuration.
                results_by_key[key] = row

                # Sauvegarder immédiatement tous les résultats.
                save_results(results_by_key)

                successful_count = sum(
                    result.get("status") == "success"
                    for result in results_by_key.values()
                )

                print(
                    f"\nProgression : "
                    f"{len(results_by_key)}/{total} configurations enregistrées"
                )

                print(
                    f"Configurations réussies : "
                    f"{successful_count}/{total}"
                )

    # Sauvegarde finale dans l'ordre du protocole.
    save_results(results_by_key)

    final_rows = sort_results(results_by_key)

    successful_count = sum(
        row.get("status") == "success"
        for row in final_rows
    )

    error_count = sum(
        str(row.get("status", "")).startswith("error")
        for row in final_rows
    )

    missing_count = total - len(final_rows)

    print("\n" + "=" * 60)
    print("ÉVALUATION TERMINÉE")
    print("=" * 60)
    print(f"Configurations réussies : {successful_count}/{total}")
    print(f"Configurations en erreur : {error_count}")
    print(f"Configurations manquantes : {missing_count}")
    print(f"Résultats enregistrés dans : {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
