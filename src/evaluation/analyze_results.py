
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "results" / "lighting_evaluation.csv"

ANALYSIS_DIR = PROJECT_ROOT / "results" / "analysis"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

CATEGORIES = ["metal_nut", "screw", "transistor"]

SHIFTS = [
    "brightness",
    "contrast",
    "gamma",
    "color_temperature",
    "shading",
]

SEVERITIES = [1, 2, 3]

KEY_COLUMNS = ["category", "shift", "severity"]

METRIC_NAMES = [
    "image_AUROC",
    "image_F1Score",
    "pixel_AUROC",
    "pixel_F1Score",
]


# ============================================================
# PREPARATION DES DOSSIERS
# ============================================================

def prepare_directories():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CHARGEMENT ET VERIFICATION DES DONNEES
# ============================================================

def load_results():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Fichier de résultats introuvable : {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = KEY_COLUMNS + METRIC_NAMES + ["status"]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes dans le CSV : {missing_columns}"
        )

    df["category"] = df["category"].astype(str)
    df["shift"] = df["shift"].astype(str)
    df["severity"] = pd.to_numeric(
        df["severity"], errors="coerce"
    )

    df["status"] = (
        df["status"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    for metric in METRIC_NAMES:
        df[metric] = pd.to_numeric(
            df[metric], errors="coerce"
        )

    # Conserver uniquement les configurations du protocole.
    success = df[df["status"] == "success"].copy()

    success = success.dropna(subset=KEY_COLUMNS)

    success["severity"] = success["severity"].astype(int)

    success = success[
        success["category"].isin(CATEGORIES)
        & success["shift"].isin(SHIFTS)
        & success["severity"].isin(SEVERITIES)
    ]

    # Détecter les doublons avant de les éliminer.
    duplicate_count = int(
        success.duplicated(KEY_COLUMNS).sum()
    )

    success = success.drop_duplicates(
        subset=KEY_COLUMNS,
        keep="last",
    ).copy()

    return df, success, duplicate_count


# ============================================================
# VERIFICATION DES 45 CONFIGURATIONS
# ============================================================

def generate_expected_configurations():

    return pd.DataFrame(
        list(product(CATEGORIES, SHIFTS, SEVERITIES)),
        columns=KEY_COLUMNS,
    )


def check_completeness(df, success, duplicate_count):

    expected = generate_expected_configurations()

    observed = success[KEY_COLUMNS].drop_duplicates()

    missing = expected.merge(
        observed,
        on=KEY_COLUMNS,
        how="left",
        indicator=True,
    )

    missing = missing[
        missing["_merge"] == "left_only"
    ].drop(columns=["_merge"])

    unsuccessful = df[
        df["status"] != "success"
    ].copy()

    # Sauvegarder les configurations manquantes et en erreur.
    missing.to_csv(
        ANALYSIS_DIR / "missing_configurations.csv",
        index=False,
    )

    unsuccessful.to_csv(
        ANALYSIS_DIR / "unsuccessful_configurations.csv",
        index=False,
    )

    missing_metric_cells = int(
        success[METRIC_NAMES].isna().sum().sum()
    )

    out_of_range = int(
        (
            (success[METRIC_NAMES] < 0)
            | (success[METRIC_NAMES] > 1)
        ).sum().sum()
    )

    report = [
        "PATCHCORE LIGHTING EVALUATION REPORT",
        "=" * 45,
        f"Expected configurations: {len(expected)}",
        f"Successful configurations: {len(success)}",
        f"Missing configurations: {len(missing)}",
        f"Unsuccessful CSV rows: {len(unsuccessful)}",
        f"Duplicate successful rows: {duplicate_count}",
        f"Missing metric values: {missing_metric_cells}",
        f"Metric values outside [0, 1]: {out_of_range}",
        "",
        "MISSING CONFIGURATIONS:",
    ]

    if missing.empty:
        report.append("None")
    else:
        for _, row in missing.iterrows():
            report.append(
                f"- {row['category']} / {row['shift']} "
                f"/ severity {row['severity']}"
            )

    report_path = ANALYSIS_DIR / "evaluation_report.txt"

    report_path.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("\n".join(report))

    return missing


# ============================================================
# TABLEAUX RECAPITULATIFS
# ============================================================

def generate_summaries(success):

    if success.empty:
        print("Aucun résultat réussi à analyser.")
        return

    # Moyennes par catégorie et transformation.
    by_category_shift = (
        success
        .groupby(["category", "shift"])[METRIC_NAMES]
        .mean()
        .reset_index()
    )

    by_category_shift.to_csv(
        ANALYSIS_DIR / "summary_by_category_shift.csv",
        index=False,
    )

    # Moyennes globales par transformation.
    by_shift = (
        success
        .groupby("shift")[METRIC_NAMES]
        .mean()
        .reindex(SHIFTS)
        .reset_index()
    )

    by_shift.to_csv(
        ANALYSIS_DIR / "summary_by_shift.csv",
        index=False,
    )

    # Moyennes par transformation et sévérité.
    by_severity = (
        success
        .groupby(["shift", "severity"])[METRIC_NAMES]
        .mean()
        .reset_index()
    )

    by_severity.to_csv(
        ANALYSIS_DIR / "summary_by_shift_severity.csv",
        index=False,
    )

    # Résultats détaillés des configurations réussies.
    success.sort_values(
        ["category", "shift", "severity"]
    ).to_csv(
        ANALYSIS_DIR / "successful_configurations.csv",
        index=False,
    )

    print("\nTableaux récapitulatifs générés.")


# ============================================================
# GRAPHIQUES : MOYENNES PAR TRANSFORMATION
# ============================================================

def plot_metric_by_shift(success, metric):

    means = (
        success
        .groupby("shift")[metric]
        .mean()
        .reindex(SHIFTS)
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(
        SHIFTS,
        means.values,
    )

    ax.set_title(f"{metric} by Lighting Transformation")
    ax.set_xlabel("Lighting Transformation")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1.05)

    ax.tick_params(axis="x", rotation=20)

    for bar, value in zip(bars, means.values):
        if pd.notna(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015,
                f"{value:.3f}",
                ha="center",
                fontsize=9,
            )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / f"{metric}_by_shift.png",
        dpi=300,
    )

    plt.close(fig)


# ============================================================
# GRAPHIQUES : INFLUENCE DE LA SEVERITE
# ============================================================

def plot_metric_by_severity(success, metric):

    fig, ax = plt.subplots(figsize=(10, 5))

    for shift in SHIFTS:

        subset = success[
            success["shift"] == shift
        ]

        means = (
            subset
            .groupby("severity")[metric]
            .mean()
            .reindex(SEVERITIES)
        )

        ax.plot(
            SEVERITIES,
            means.values,
            marker="o",
            label=shift,
        )

    ax.set_title(f"{metric} by Lighting Severity")
    ax.set_xlabel("Severity Level")
    ax.set_ylabel(metric)
    ax.set_xticks(SEVERITIES)
    ax.set_ylim(0, 1.05)
    ax.legend(title="Transformation")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / f"{metric}_by_severity.png",
        dpi=300,
    )

    plt.close(fig)


# ============================================================
# GRAPHIQUES : COMPARAISON DES CATEGORIES
# ============================================================

def plot_category_heatmap(success, metric):

    pivot = success.pivot_table(
        index="category",
        columns="shift",
        values=metric,
        aggfunc="mean",
    )

    pivot = pivot.reindex(
        index=CATEGORIES,
        columns=SHIFTS,
    )

    values = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(11, 5))

    image = ax.imshow(
        values,
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    ax.set_xticks(range(len(SHIFTS)))
    ax.set_xticklabels(SHIFTS, rotation=20)

    ax.set_yticks(range(len(CATEGORIES)))
    ax.set_yticklabels(CATEGORIES)

    ax.set_title(f"{metric}: Category Comparison")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):

            value = values[i, j]

            if np.isfinite(value):
                ax.text(
                    j,
                    i,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                )

    fig.colorbar(
        image,
        ax=ax,
        label=metric,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / f"{metric}_category_heatmap.png",
        dpi=300,
    )

    plt.close(fig)


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    prepare_directories()

    print("\nANALYSE DES RESULTATS PATCHCORE")
    print("=" * 50)

    df, success, duplicate_count = load_results()

    missing = check_completeness(
        df,
        success,
        duplicate_count,
    )

    generate_summaries(success)

    if not success.empty:

        for metric in METRIC_NAMES:

            plot_metric_by_shift(
                success,
                metric,
            )

            plot_metric_by_severity(
                success,
                metric,
            )

            plot_category_heatmap(
                success,
                metric,
            )

    print("\nANALYSE TERMINEE")
    print(f"Configurations réussies : {len(success)}/45")
    print(f"Configurations manquantes : {len(missing)}")
    print(f"Tableaux : {ANALYSIS_DIR}")
    print(f"Graphiques : {FIGURES_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
