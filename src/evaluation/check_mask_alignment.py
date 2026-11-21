
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"

CATEGORIES = ["metal_nut", "screw", "transistor"]

SHIFTS = [
    "brightness",
    "contrast",
    "gamma",
    "color_temperature",
    "shading",
]

SEVERITIES = [1, 2, 3]

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}


def get_files_in_loader_order(folder):
    """Reproduit l'ordre de parcours récursif utilisé par Folder."""
    return [
        path
        for path in folder.glob("**/*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def main():

    total_checks = 0
    successful_checks = 0
    problems = []

    print("\nVERIFICATION DE L'ALIGNEMENT DES MASQUES")
    print("=" * 60)

    for category in CATEGORIES:
        for shift in SHIFTS:
            for severity in SEVERITIES:

                test_root = (
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

                defect_dirs = sorted(
                    path
                    for path in test_root.iterdir()
                    if path.is_dir() and path.name != "good"
                )

                for defect_dir in defect_dirs:

                    mask_dir = masks_root / defect_dir.name

                    total_checks += 1

                    if not mask_dir.exists():
                        problems.append(
                            f"Missing mask folder: {category}/{shift}/"
                            f"severity_{severity}/{defect_dir.name}"
                        )
                        continue

                    image_files = get_files_in_loader_order(defect_dir)
                    mask_files = get_files_in_loader_order(mask_dir)

                    image_names = [p.name for p in image_files]
                    mask_names = [p.name for p in mask_files]

                    if image_names == mask_names:
                        successful_checks += 1
                    else:
                        problems.append(
                            f"File/order mismatch: {category}/{shift}/"
                            f"severity_{severity}/{defect_dir.name}"
                        )

    print(f"Folders vérifiés : {total_checks}")
    print(f"Correspondances exactes : {successful_checks}")
    print(f"Problèmes détectés : {len(problems)}")

    for problem in problems:
        print("PROBLEM:", problem)

    if not problems:
        print("\nSUCCESS: Image/mask filenames and order match.")
    else:
        print("\nWARNING: Some image/mask folders need investigation.")

    print("=" * 60)


if __name__ == "__main__":
    main()
