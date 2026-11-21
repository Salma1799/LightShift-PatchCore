
from pathlib import Path
import cv2

DATASET_ROOT = Path("data/raw/mvtec_anomaly_detection")
PROCESSED_ROOT = Path("data/processed")

CATEGORIES = ["metal_nut", "screw", "transistor"]
SHIFTS = [
    "brightness",
    "contrast",
    "gamma",
    "color_temperature",
    "shading",
]
SEVERITIES = [1, 2, 3]


def copy_category_masks(category):
    category_root = DATASET_ROOT / category
    processed_root = PROCESSED_ROOT / category

    total_masks = 0

    for shift in SHIFTS:
        for severity in SEVERITIES:

            transformed_root = (
                processed_root
                / "test"
                / shift
                / f"severity_{severity}"
            )

            mask_root = (
                processed_root
                / "masks"
                / shift
                / f"severity_{severity}"
            )

            if not transformed_root.exists():
                raise FileNotFoundError(
                    f"Missing transformed folder: {transformed_root}"
                )

            for defect_folder in transformed_root.iterdir():
                if not defect_folder.is_dir():
                    continue

                defect = defect_folder.name

                for image_path in defect_folder.glob("*.png"):

                    if defect == "good":
                        original_image = cv2.imread(
                            str(image_path)
                        )

                        if original_image is None:
                            raise ValueError(
                                f"Cannot read image: {image_path}"
                            )

                        mask = original_image[:, :, 0] * 0

                    else:
                        original_mask_path = (
                            category_root
                            / "ground_truth"
                            / defect
                            / f"{image_path.stem}_mask.png"
                        )

                        if not original_mask_path.exists():
                            raise FileNotFoundError(
                                f"Mask not found: {original_mask_path}"
                            )

                        mask = cv2.imread(
                            str(original_mask_path),
                            cv2.IMREAD_GRAYSCALE
                        )

                        if mask is None:
                            raise ValueError(
                                f"Cannot read mask: {original_mask_path}"
                            )

                    output_dir = mask_root / defect
                    output_dir.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    output_path = output_dir / image_path.name

                    if not cv2.imwrite(str(output_path), mask):
                        raise RuntimeError(
                            f"Cannot save mask: {output_path}"
                        )

                    total_masks += 1

    print(f"{category}: {total_masks} masks generated")


if __name__ == "__main__":
    for category in CATEGORIES:
        copy_category_masks(category)

    print("All masks generated successfully.")
