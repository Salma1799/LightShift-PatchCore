
from pathlib import Path
import argparse

from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine


DATASET_ROOT = Path("data/raw/mvtec_anomaly_detection")
RESULTS_ROOT = Path("results/Patchcore/MVTecAD")

CATEGORIES = ["metal_nut", "screw", "transistor"]


def train_category(category):

    checkpoint = (
        RESULTS_ROOT
        / category
        / "latest"
        / "weights"
        / "lightning"
        / "model.ckpt"
    )

    if checkpoint.exists():
        print(f"Un checkpoint existe déjà pour {category}.")
        print(f"Checkpoint : {checkpoint}")
        print("Entraînement ignoré.")
        return

    print(f"\nChargement de la catégorie : {category}")

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
        max_epochs=1,
        accelerator="cpu",
        devices=1,
    )

    print(f"Début de l'entraînement : {category}")

    engine.fit(
        model=model,
        datamodule=datamodule,
    )

    print(f"Évaluation sur le dataset original : {category}")

    results = engine.test(
        model=model,
        datamodule=datamodule,
    )

    print(f"Résultats pour {category} :")
    print(results)

    if checkpoint.exists():
        print(f"Checkpoint sauvegardé : {checkpoint}")
    else:
        print(
            "Attention : vérifie le dossier des résultats, "
            "le checkpoint attendu n'a pas été trouvé."
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--category",
        choices=CATEGORIES,
        required=True,
        help="Catégorie MVTec AD à entraîner",
    )

    args = parser.parse_args()

    train_category(args.category)
