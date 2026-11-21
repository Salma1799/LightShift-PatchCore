
from pathlib import Path
import cv2
import numpy as np

# Configuration
DATASET_ROOT = Path("data/raw/mvtec_anomaly_detection")
OUTPUT_ROOT = Path("data/processed")

CATEGORIES = ["metal_nut", "screw", "transistor"]

SEVERITIES = [1, 2, 3]

SHIFTS = [
    "brightness",
    "contrast",
    "gamma",
    "color_temperature",
    "shading",
]


def apply_shift(image, shift, severity):
    """Applique une perturbation photométrique à une image."""

    image = image.astype(np.float32) / 255.0

    if shift == "brightness":
        factors = {1: 0.8, 2: 0.6, 3: 0.4}
        result = image * factors[severity]

    elif shift == "contrast":
        factors = {1: 0.8, 2: 0.6, 3: 0.4}
        factor = factors[severity]
        result = (image - 0.5) * factor + 0.5

    elif shift == "gamma":
        gammas = {1: 0.7, 2: 1.5, 3: 2.0}
        result = np.power(np.clip(image, 0, 1), gammas[severity])

    elif shift == "color_temperature":
        factors = {
            1: (1.10, 1.0, 0.90),
            2: (1.20, 1.0, 0.80),
            3: (1.30, 1.0, 0.70),
        }
        red, green, blue = factors[severity]
        result = image.copy()
        result[:, :, 2] *= red
        result[:, :, 1] *= green
        result[:, :, 0] *= blue

    elif shift == "shading":
        h, w = image.shape[:2]
        strength = {1: 0.2, 2: 0.4, 3: 0.6}[severity]

        gradient = np.linspace(
            1.0 - strength, 1.0, w, dtype=np.float32
        )
        mask = np.tile(gradient, (h, 1))
        result = image * mask[:, :, np.newaxis]

    else:
        raise ValueError(f"Shift inconnu : {shift}")

    return np.clip(result * 255, 0, 255).astype(np.uint8)


def process_category(category):
    category_root = DATASET_ROOT / category

    for split in ["train", "test"]:
        split_root = category_root / split

        for defect_folder in split_root.iterdir():
            if not defect_folder.is_dir():
                continue

            for image_path in defect_folder.glob("*.png"):
                image = cv2.imread(str(image_path))

                if image is None:
                    print(f"Image ignorée : {image_path}")
                    continue

                for shift in SHIFTS:
                    for severity in SEVERITIES:

                        output_dir = (
                            OUTPUT_ROOT
                            / category
                            / split
                            / shift
                            / f"severity_{severity}"
                            / defect_folder.name
                        )

                        output_dir.mkdir(
                            parents=True,
                            exist_ok=True
                        )

                        transformed = apply_shift(
                            image, shift, severity
                        )

                        output_path = output_dir / image_path.name

                        cv2.imwrite(
                            str(output_path),
                            transformed
                        )

    print(f"Génération terminée : {category}")


if __name__ == "__main__":
    for category in CATEGORIES:
        process_category(category)

    print("Toutes les variations d'éclairage sont générées.")
