
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASELINE_FILE = PROJECT_ROOT / "results" / "baseline_metrics.csv"
LIGHTING_FILE = PROJECT_ROOT / "results" / "lighting_evaluation.csv"
OUTPUT_FILE = (
    PROJECT_ROOT / "results" / "analysis"
    / "baseline_vs_lighting.csv"
)

METRICS = [
    "image_AUROC",
    "image_F1Score",
    "pixel_AUROC",
    "pixel_F1Score",
]


def main():

    baseline = pd.read_csv(BASELINE_FILE)
    lighting = pd.read_csv(LIGHTING_FILE)

    if not (baseline["status"] == "success").all():
        raise ValueError("Certaines références originales sont en erreur.")

    successful = lighting[
        lighting["status"] == "success"
    ].copy()

    if len(successful) != 45:
        raise ValueError(
            f"45 configurations attendues, "
            f"{len(successful)} trouvées."
        )

    # Moyenne des 15 configurations d'éclairage par catégorie.
    lighting_means = (
        successful.groupby("category")[METRICS]
        .mean()
        .reset_index()
    )

    comparison = baseline[
        ["category", *METRICS]
    ].merge(
        lighting_means,
        on="category",
        suffixes=("_original", "_lighting_mean"),
    )

    # Calcul des variations absolues et relatives.
    for metric in METRICS:

        original = comparison[f"{metric}_original"]
        modified = comparison[f"{metric}_lighting_mean"]

        comparison[f"{metric}_delta"] = modified - original

        comparison[f"{metric}_change_pct"] = (
            (modified - original) / original
        ) * 100

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        OUTPUT_FILE,
        index=False,
        float_format="%.6f",
    )

    print("\nBASELINE VS LIGHTING COMPARISON")
    print("=" * 55)

    for _, row in comparison.iterrows():

        print(f"\nCategory: {row['category']}")

        for metric in METRICS:

            print(
                f"{metric}: "
                f"{row[f'{metric}_original']:.4f} -> "
                f"{row[f'{metric}_lighting_mean']:.4f} "
                f"({row[f'{metric}_change_pct']:+.2f}%)"
            )

    print("\nComparison saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
