"""
Statistical analysis, visualization, and reporting for biomarker validation.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from biomarker_validation.constants import (
    BIOMARKER_COLUMNS,
    DEFAULT_RANDOM_SEED,
    DISEASE_LABELS,
    STAGE_NAMES,
)

sns.set_theme(style="whitegrid")


def load_results_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in BIOMARKER_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _describe_series(series: pd.Series) -> Dict[str, Any]:
    valid = series.dropna()
    return {
        "count": int(valid.count()),
        "missing_count": int(series.isna().sum()),
        "missing_pct": round(float(series.isna().mean() * 100), 2),
        "min": round(float(valid.min()), 4) if len(valid) else None,
        "max": round(float(valid.max()), 4) if len(valid) else None,
        "mean": round(float(valid.mean()), 4) if len(valid) else None,
        "median": round(float(valid.median()), 4) if len(valid) else None,
        "std": round(float(valid.std(ddof=1)), 4) if len(valid) > 1 else None,
    }


def compute_biomarker_statistics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(df)
    success = int((df["extraction_status"] == "success").sum()) if "extraction_status" in df.columns else total
    failed = total - success

    rows.append(
        {
            "metric": "total_images_processed",
            "value": total,
            "biomarker": "ALL",
        }
    )
    rows.append({"metric": "successful_extractions", "value": success, "biomarker": "ALL"})
    rows.append({"metric": "failed_extractions", "value": failed, "biomarker": "ALL"})

    for biomarker in BIOMARKER_COLUMNS:
        stats_dict = _describe_series(df[biomarker])
        for key, value in stats_dict.items():
            rows.append(
                {
                    "metric": key,
                    "value": value,
                    "biomarker": biomarker,
                }
            )
    return pd.DataFrame(rows)


def compute_group_statistics(
    df: pd.DataFrame,
    group_col: str,
    groups: List[str],
) -> pd.DataFrame:
    rows = []
    for biomarker in BIOMARKER_COLUMNS:
        for group in groups:
            subset = df[df[group_col] == group][biomarker]
            desc = _describe_series(subset)
            rows.append(
                {
                    "biomarker": biomarker,
                    group_col: group,
                    **desc,
                }
            )
    return pd.DataFrame(rows)


def compute_stage_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute biomarker stats by dataset_folder and stage."""
    rows = []
    staged = df[df["stage"].isin(STAGE_NAMES)]
    for biomarker in BIOMARKER_COLUMNS:
        for dataset_folder in sorted(staged["dataset_folder"].unique()):
            folder_df = staged[staged["dataset_folder"] == dataset_folder]
            mapped_disease = folder_df["disease"].iloc[0]
            for stage in STAGE_NAMES:
                subset = folder_df[folder_df["stage"] == stage][biomarker]
                desc = _describe_series(subset)
                rows.append(
                    {
                        "biomarker": biomarker,
                        "dataset_folder": dataset_folder,
                        "disease": mapped_disease,
                        "stage": stage,
                        **desc,
                    }
                )
    return pd.DataFrame(rows)


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    corr = df[BIOMARKER_COLUMNS].corr(method="pearson", min_periods=3)
    return corr.round(4)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if len(a) < 2 or len(b) < 2:
        return None
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled = np.sqrt(((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def compute_separability_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kruskal-Wallis across diseases and pairwise Healthy-vs-disease effect sizes.
    """
    rows = []
    healthy = df[df["disease"] == "Healthy"]

    for biomarker in BIOMARKER_COLUMNS:
        groups = [
            df[df["disease"] == disease][biomarker].dropna().values
            for disease in DISEASE_LABELS
        ]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
            h_stat, p_value = stats.kruskal(*groups)
            rows.append(
                {
                    "biomarker": biomarker,
                    "test_name": "Kruskal-Wallis",
                    "comparison": "Across all diseases",
                    "statistic": round(float(h_stat), 4),
                    "p_value": float(p_value),
                    "interpretation": "Tests whether biomarker distributions differ across disease groups (non-parametric).",
                }
            )

        # One-way ANOVA reported with caution (assumes approximate normality / large-sample robustness)
        anova_groups = [
            df[df["disease"] == disease][biomarker].dropna().values
            for disease in DISEASE_LABELS
        ]
        anova_groups = [g for g in anova_groups if len(g) > 0]
        if len(anova_groups) >= 2:
            f_stat, p_value = stats.f_oneway(*anova_groups)
            rows.append(
                {
                    "biomarker": biomarker,
                    "test_name": "One-way ANOVA",
                    "comparison": "Across all diseases",
                    "statistic": round(float(f_stat), 4),
                    "p_value": float(p_value),
                    "interpretation": "Parametric test for mean differences; interpret cautiously if distributions are skewed.",
                }
            )

        healthy_vals = healthy[biomarker].dropna().values
        for disease in DISEASE_LABELS:
            if disease == "Healthy":
                continue
            disease_vals = df[df["disease"] == disease][biomarker].dropna().values
            if len(healthy_vals) < 2 or len(disease_vals) < 2:
                continue
            u_stat, p_value = stats.mannwhitneyu(healthy_vals, disease_vals, alternative="two-sided")
            d = _cohens_d(healthy_vals, disease_vals)
            rows.append(
                {
                    "biomarker": biomarker,
                    "test_name": "Mann-Whitney U",
                    "comparison": f"Healthy vs {disease}",
                    "statistic": round(float(u_stat), 4),
                    "p_value": float(p_value),
                    "effect_size_name": "Cohen's d",
                    "effect_size_value": round(d, 4) if d is not None else None,
                    "interpretation": "Cohen's d measures standardized mean difference; |d|≈0.2 small, 0.5 medium, 0.8 large.",
                }
            )
    return pd.DataFrame(rows)


def compute_stage_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    staged = df[df["stage"].isin(STAGE_NAMES)]
    for dataset_folder in sorted(staged["dataset_folder"].unique()):
        folder_df = staged[staged["dataset_folder"] == dataset_folder]
        for biomarker in BIOMARKER_COLUMNS:
            groups = [
                folder_df[folder_df["stage"] == stage][biomarker].dropna().values
                for stage in STAGE_NAMES
            ]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) >= 2:
                h_stat, p_value = stats.kruskal(*groups)
                rows.append(
                    {
                        "dataset_folder": dataset_folder,
                        "mapped_disease": folder_df["disease"].iloc[0],
                        "biomarker": biomarker,
                        "test_name": "Kruskal-Wallis",
                        "comparison": "Across stages (early/moderate/severe)",
                        "statistic": round(float(h_stat), 4),
                        "p_value": float(p_value),
                    }
                )
    return pd.DataFrame(rows)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_disease_boxplots(df: pd.DataFrame, output_dir: str) -> None:
    _ensure_dir(output_dir)
    for biomarker in BIOMARKER_COLUMNS:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="disease", y=biomarker, order=DISEASE_LABELS)
        plt.title(f"{biomarker.replace('_', ' ').title()} by Disease")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{biomarker}_by_disease.png"), dpi=150)
        plt.close()


def plot_stage_boxplots(df: pd.DataFrame, output_dir: str) -> None:
    _ensure_dir(output_dir)
    staged = df[df["stage"].isin(STAGE_NAMES)]
    for dataset_folder in sorted(staged["dataset_folder"].unique()):
        folder_df = staged[staged["dataset_folder"] == dataset_folder]
        for biomarker in BIOMARKER_COLUMNS:
            plt.figure(figsize=(7, 5))
            sns.boxplot(
                data=folder_df,
                x="stage",
                y=biomarker,
                order=STAGE_NAMES,
            )
            plt.title(
                f"{biomarker.replace('_', ' ').title()} by Stage ({dataset_folder} -> {folder_df['disease'].iloc[0]})"
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"{dataset_folder}_{biomarker}_by_stage.png"),
                dpi=150,
            )
            plt.close()


def plot_mean_comparisons(disease_stats: pd.DataFrame, output_dir: str) -> None:
    _ensure_dir(output_dir)
    means = disease_stats.pivot(index="disease", columns="biomarker", values="mean")
    means = means.reindex(DISEASE_LABELS)
    plt.figure(figsize=(10, 6))
    sns.heatmap(means.astype(float), annot=True, fmt=".3f", cmap="YlOrRd")
    plt.title("Disease-wise Mean Biomarker Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "disease_mean_heatmap.png"), dpi=150)
    plt.close()


def plot_missing_summary(df: pd.DataFrame, output_dir: str) -> None:
    _ensure_dir(output_dir)
    missing_pct = df[BIOMARKER_COLUMNS].isna().mean() * 100
    plt.figure(figsize=(8, 5))
    sns.barplot(x=missing_pct.index, y=missing_pct.values)
    plt.ylabel("Missing (%)")
    plt.title("Missing Biomarker Values")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "missing_value_summary.png"), dpi=150)
    plt.close()


def plot_correlation_heatmap(corr: pd.DataFrame, output_dir: str) -> None:
    _ensure_dir(output_dir)
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Biomarker Correlation Matrix (Pearson)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "biomarker_correlation_heatmap.png"), dpi=150)
    plt.close()


def run_baseline_experiment(
    df: pd.DataFrame,
    output_dir: str,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Isolated baseline: 5 biomarkers -> disease category.
    Uses median imputation for missing biomarkers and reports limitations.
    """
    _ensure_dir(output_dir)
    model_df = df[df["extraction_status"] == "success"].copy()
    complete_before = len(model_df)
    model_df = model_df.dropna(subset=["disease"])
    complete_cases = model_df.dropna(subset=BIOMARKER_COLUMNS)
    complete_case_count = len(complete_cases)

    X = model_df[BIOMARKER_COLUMNS]
    y = model_df["disease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_seed,
        stratify=y if y.nunique() > 1 else None,
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=random_seed,
                    multi_class="multinomial",
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, zero_division=0)

    metrics_path = os.path.join(output_dir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        handle.write("Optional Baseline Classification Experiment\n")
        handle.write("===========================================\n\n")
        handle.write("Input features: 5 visual biomarkers\n")
        handle.write("Target: mapped disease label\n")
        handle.write("Model: Logistic Regression (median imputation + standardization)\n")
        handle.write(f"Random seed: {random_seed}\n")
        handle.write(f"Train size: {len(X_train)}\n")
        handle.write(f"Test size: {len(X_test)}\n")
        handle.write(f"Rows with all biomarkers present: {complete_case_count} / {complete_before}\n\n")
        handle.write(f"Accuracy: {accuracy:.4f}\n")
        handle.write(f"Macro F1: {macro_f1:.4f}\n\n")
        handle.write("Classification Report:\n")
        handle.write(report)
        handle.write("\nLimitations:\n")
        handle.write("- This is NOT the production MobileNet + Random Forest classifier.\n")
        handle.write("- Missing biomarkers were median-imputed, which may bias results.\n")
        handle.write("- Class imbalance and folder mapping (foot/mouth) may affect metrics.\n")
        handle.write("- Statistical performance here does not imply clinical usefulness.\n")

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=DISEASE_LABELS,
        xticks_rotation=25,
        ax=ax,
    )
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()

    return {
        "performed": True,
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "metrics_path": metrics_path,
        "confusion_matrix_path": cm_path,
        "complete_cases": complete_case_count,
        "total_success_rows": complete_before,
    }


def assess_biomarker_value(
    biomarker_stats: pd.DataFrame,
    disease_stats: pd.DataFrame,
    separability: pd.DataFrame,
    corr: pd.DataFrame,
    stage_stats: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for biomarker in BIOMARKER_COLUMNS:
        bstats = biomarker_stats[
            (biomarker_stats["biomarker"] == biomarker) & (biomarker_stats["metric"] == "missing_pct")
        ]
        missing_pct = float(bstats["value"].iloc[0]) if len(bstats) else None

        kw = separability[
            (separability["biomarker"] == biomarker)
            & (separability["test_name"] == "Kruskal-Wallis")
        ]
        kw_p = float(kw["p_value"].iloc[0]) if len(kw) else None

        if "effect_size_name" in separability.columns:
            effects = separability[
                (separability["biomarker"] == biomarker)
                & (separability["effect_size_name"] == "Cohen's d")
            ]
        else:
            effects = separability.iloc[0:0]
        max_abs_d = (
            float(effects["effect_size_value"].abs().max())
            if len(effects) and effects["effect_size_value"].notna().any()
            else None
        )

        high_corr = []
        for other in BIOMARKER_COLUMNS:
            if other == biomarker:
                continue
            val = corr.loc[biomarker, other]
            if pd.notna(val) and abs(val) >= 0.7:
                high_corr.append(f"{other} ({val:.2f})")

        stage_rows = stage_stats[stage_stats["biomarker"] == biomarker]
        stage_means = stage_rows.groupby("stage")["mean"].mean().dropna()
        stage_pattern = "insufficient staged data"
        if len(stage_means) >= 2:
            stage_pattern = ", ".join(f"{k}={v:.3f}" for k, v in stage_means.items())

        recommendation = "Insufficient evidence"
        evidence_parts = []
        if missing_pct is not None:
            evidence_parts.append(f"missing={missing_pct:.1f}%")
        if kw_p is not None:
            evidence_parts.append(f"Kruskal-Wallis p={kw_p:.4g}")
        if max_abs_d is not None:
            evidence_parts.append(f"max |Cohen's d|={max_abs_d:.3f}")
        if high_corr:
            evidence_parts.append("high correlation with " + "; ".join(high_corr))

        if missing_pct is not None and missing_pct > 20:
            recommendation = "Currently unreliable"
        elif kw_p is not None and kw_p < 0.05 and max_abs_d is not None and max_abs_d >= 0.3:
            recommendation = "Potentially useful"
        elif kw_p is not None and kw_p < 0.05:
            recommendation = "Needs further validation"
        elif missing_pct is not None and missing_pct <= 5:
            recommendation = "Needs further validation"

        rows.append(
            {
                "biomarker": biomarker,
                "missing_pct": missing_pct,
                "kruskal_wallis_p": kw_p,
                "max_abs_cohens_d_vs_healthy": max_abs_d,
                "high_correlations": "; ".join(high_corr) if high_corr else "",
                "stage_mean_pattern": stage_pattern,
                "evidence_summary": "; ".join(evidence_parts),
                "recommendation": recommendation,
            }
        )
    return pd.DataFrame(rows)


def generate_analysis_report(
    output_path: str,
    discovery: Dict[str, Any],
    extraction_meta: Dict[str, Any],
    biomarker_stats: pd.DataFrame,
    disease_stats: pd.DataFrame,
    stage_stats: pd.DataFrame,
    corr: pd.DataFrame,
    separability: pd.DataFrame,
    stage_tests: pd.DataFrame,
    biomarker_assessment: pd.DataFrame,
    baseline_result: Dict[str, Any],
) -> None:
    lines: List[str] = []
    lines.append("# KrishiRakshak Biomarker Validation Report")
    lines.append("")
    lines.append("Experimental analysis of image-derived visual biomarkers. Not medical validation.")
    lines.append("")

    lines.append("## 1. Dataset Used")
    lines.append(f"- Root: `{discovery['dataset_root']}`")
    lines.append("- This is the dataset used by the current project (`app.py`, training scripts).")
    lines.append("- **Foot and Mouth mapping:** `dataset/foot/` and `dataset/mouth/` are preserved as separate folders.")
    lines.append("  Grouped disease analysis uses the label **Foot and Mouth** for both.")
    lines.append("")

    lines.append("## 2. Dataset Size")
    lines.append(f"- Total images in dataset: {discovery['total_images']}")
    lines.append(f"- Images processed in this run: {extraction_meta['images_selected']}")
    lines.append(f"- Successful extractions: {extraction_meta['successful_extractions']}")
    lines.append(f"- Failed extractions: {extraction_meta['failed_extractions']}")
    lines.append("")

    lines.append("### Images by dataset folder")
    for folder, count in discovery.get("images_by_folder", {}).items():
        mapped = discovery["categories"][folder]["mapped_disease"]
        lines.append(f"- `{folder}` -> {mapped}: {count}")
    lines.append("")

    lines.append("## 3. Sampling Method")
    lines.append(f"- Mode: **{extraction_meta['sampling_mode']}**")
    if extraction_meta.get("max_per_disease") is not None:
        lines.append(f"- `--max-per-class`: {extraction_meta['max_per_disease']} per mapped disease label")
        lines.append(f"- Random seed: {extraction_meta['random_seed']}")
    else:
        lines.append("- Full dataset processed (no sampling).")
    lines.append("")

    lines.append("## 4. Biomarker Extraction Method")
    lines.append("Existing `biomarker_extraction/` package (unchanged):")
    lines.append("- Lesion Density: lesion pixel ratio")
    lines.append("- Texture Roughness: uniform LBP + Sobel gradient")
    lines.append("- Color Variation: HSV S/V standard deviation")
    lines.append("- Lesion Clustering: connected-component centroid distances")
    lines.append("- Boundary Irregularity: contour circularity deviation")
    lines.append("")

    lines.append("## 5. Data Quality Results")
    for biomarker in BIOMARKER_COLUMNS:
        subset = biomarker_stats[biomarker_stats["biomarker"] == biomarker]
        mean_row = subset[subset["metric"] == "mean"]
        missing_row = subset[subset["metric"] == "missing_pct"]
        if len(mean_row):
            lines.append(
                f"- **{biomarker}**: mean={mean_row['value'].iloc[0]}, "
                f"missing={missing_row['value'].iloc[0]}%"
            )
    lines.append("")

    lines.append("## 6. Disease-level Statistics")
    lines.append("See `disease_statistics.csv` for full tables.")
    lines.append("")
    for biomarker in BIOMARKER_COLUMNS:
        lines.append(f"### {biomarker}")
        sub = disease_stats[disease_stats["biomarker"] == biomarker]
        for _, row in sub.iterrows():
            lines.append(
                f"- **{row['disease']}** (n={row['count']}): "
                f"mean={row['mean']}, median={row['median']}, std={row['std']}, "
                f"min={row['min']}, max={row['max']}"
            )
        lines.append("")

    lines.append("## 7. Stage-level Statistics")
    lines.append("See `stage_statistics.csv`. Stage analysis preserves `dataset_folder`.")
    lines.append("")

    lines.append("## 8. Missing-value Analysis")
    for biomarker in BIOMARKER_COLUMNS:
        row = biomarker_stats[
            (biomarker_stats["biomarker"] == biomarker) & (biomarker_stats["metric"] == "missing_pct")
        ]
        if len(row):
            lines.append(f"- {biomarker}: {row['value'].iloc[0]}% unavailable")
    lines.append("")

    lines.append("## 9. Distribution Observations")
    lines.append("- Box plots saved under `plots/disease_boxplots/` and `plots/stage_boxplots/`.")
    lines.append("- Disease mean heatmap saved under `plots/mean_comparisons/`.")
    lines.append("- These plots are exploratory only; no causal claims are made.")
    lines.append("")

    lines.append("## 10. Statistical Tests")
    for _, row in separability.head(20).iterrows():
        lines.append(
            f"- **{row['biomarker']}** | {row['test_name']} | {row['comparison']} | "
            f"stat={row['statistic']} | p={row['p_value']:.4g}"
        )
    lines.append("")

    lines.append("## 11. Effect-size Observations")
    if "effect_size_name" in separability.columns:
        effects = separability[separability["effect_size_name"] == "Cohen's d"]
    else:
        effects = separability.iloc[0:0]
    for _, row in effects.iterrows():
        lines.append(
            f"- {row['biomarker']} {row['comparison']}: Cohen's d={row['effect_size_value']}"
        )
    lines.append("")

    lines.append("## 12. Biomarker Correlation")
    lines.append("See `correlation_matrix.csv` and `plots/correlation/biomarker_correlation_heatmap.png`.")
    lines.append("")
    for i, a in enumerate(BIOMARKER_COLUMNS):
        for b in BIOMARKER_COLUMNS[i + 1 :]:
            val = corr.loc[a, b]
            if pd.notna(val) and abs(val) >= 0.5:
                lines.append(f"- {a} ↔ {b}: r={val:.3f}")
    lines.append("")

    lines.append("## 13. Potential Strengths")
    useful = biomarker_assessment[
        biomarker_assessment["recommendation"] == "Potentially useful"
    ]["biomarker"].tolist()
    if useful:
        for name in useful:
            lines.append(f"- {name}")
    else:
        lines.append("- No biomarker met all criteria for 'Potentially useful' in this run.")
    lines.append("")

    lines.append("## 14. Potential Weaknesses")
    weak = biomarker_assessment[
        biomarker_assessment["recommendation"].isin(
            ["Currently unreliable", "Insufficient evidence"]
        )
    ]
    for _, row in weak.iterrows():
        lines.append(f"- {row['biomarker']}: {row['recommendation']} ({row['evidence_summary']})")
    lines.append("")

    lines.append("## 15. Limitations")
    lines.append("- Segmentation may respond to background, fur, shadows, and lighting.")
    lines.append("- Color variation is scene-level and not disease-specific by itself.")
    lines.append("- Clustering/boundary biomarkers may be unavailable when too few lesion regions are detected.")
    lines.append("- foot and mouth are separate folders grouped only for analysis.")
    lines.append("- Exact duplicate images may exist; they were not removed.")
    if extraction_meta["sampling_mode"] == "sampled dataset":
        lines.append("- This run used sampled data; full-dataset statistics may differ.")
    lines.append("")

    lines.append("## 16. Recommendations for Next Research Step")
    lines.append("1. Review debug visualizations for segmentation quality on each disease folder.")
    lines.append("2. Refine lesion detection before creating disease-specific biomarker profiles.")
    lines.append("3. Proceed to Step 3: disease/stage-specific biomarker profiles using evidence from this report.")
    lines.append("4. Do not assign CSS weights until profiles are validated on held-out data.")
    lines.append("")

    lines.append("## Optional Baseline Experiment")
    if baseline_result.get("performed"):
        lines.append(f"- Performed: yes")
        lines.append(f"- Accuracy: {baseline_result['accuracy']}")
        lines.append(f"- Macro F1: {baseline_result['macro_f1']}")
        lines.append(f"- See `optional_baseline/metrics.txt`")
    else:
        lines.append("- Performed: no")
    lines.append("")

    lines.append("## Biomarker Assessment Table")
    lines.append("")
    lines.append("| Biomarker | Missing % | KW p-value | Max |d| vs Healthy | Recommendation |")
    lines.append("|-----------|-----------|------------|---------------------|----------------|")
    for _, row in biomarker_assessment.iterrows():
        lines.append(
            f"| {row['biomarker']} | {row['missing_pct']} | {row['kruskal_wallis_p']} | "
            f"{row['max_abs_cohens_d_vs_healthy']} | {row['recommendation']} |"
        )

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def run_full_analysis(
    results_csv: str,
    output_dir: str,
    discovery: Dict[str, Any],
    extraction_meta: Dict[str, Any],
    run_baseline: bool = True,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    df = load_results_csv(results_csv)

    biomarker_stats = compute_biomarker_statistics(df)
    disease_stats = compute_group_statistics(df, "disease", DISEASE_LABELS)
    stage_stats = compute_stage_statistics(df)
    corr = compute_correlation_matrix(df)
    separability = compute_separability_tests(df)
    stage_tests = compute_stage_tests(df)
    biomarker_assessment = assess_biomarker_value(
        biomarker_stats, disease_stats, separability, corr, stage_stats
    )

    os.makedirs(output_dir, exist_ok=True)
    biomarker_stats.to_csv(os.path.join(output_dir, "biomarker_statistics.csv"), index=False)
    disease_stats.to_csv(os.path.join(output_dir, "disease_statistics.csv"), index=False)
    stage_stats.to_csv(os.path.join(output_dir, "stage_statistics.csv"), index=False)
    corr.to_csv(os.path.join(output_dir, "correlation_matrix.csv"))
    separability.to_csv(os.path.join(output_dir, "separability_tests.csv"), index=False)
    stage_tests.to_csv(os.path.join(output_dir, "stage_tests.csv"), index=False)
    biomarker_assessment.to_csv(os.path.join(output_dir, "biomarker_assessment.csv"), index=False)

    plot_disease_boxplots(df, os.path.join(output_dir, "plots", "disease_boxplots"))
    plot_stage_boxplots(df, os.path.join(output_dir, "plots", "stage_boxplots"))
    plot_mean_comparisons(disease_stats, os.path.join(output_dir, "plots", "mean_comparisons"))
    plot_missing_summary(df, os.path.join(output_dir, "plots", "mean_comparisons"))
    plot_correlation_heatmap(corr, os.path.join(output_dir, "plots", "correlation"))

    baseline_result = {"performed": False}
    if run_baseline and len(df) >= 20:
        baseline_result = run_baseline_experiment(
            df,
            os.path.join(output_dir, "optional_baseline"),
            random_seed=random_seed,
        )

    generate_analysis_report(
        os.path.join(output_dir, "analysis_report.md"),
        discovery,
        extraction_meta,
        biomarker_stats,
        disease_stats,
        stage_stats,
        corr,
        separability,
        stage_tests,
        biomarker_assessment,
        baseline_result,
    )

    return {
        "biomarker_stats": biomarker_stats,
        "disease_stats": disease_stats,
        "stage_stats": stage_stats,
        "correlation": corr,
        "separability": separability,
        "biomarker_assessment": biomarker_assessment,
        "baseline_result": baseline_result,
    }
