"""
Visualization module for rent prediction model results.

This module provides comprehensive visualizations for evaluating
the XGBoost regression model performance.

Usage:
    python -m model.visualize --input data/supercasas_apartments.csv

Visualizations included:
    1. Predicted vs Actual scatter plot
    2. Residuals distribution
    3. Residuals vs Predicted values
    4. Feature importance (top 20)
    5. Error distribution by price range
    6. Prediction accuracy buckets (like confusion matrix for regression)
    7. Learning curves
    8. SHAP summary (if available)
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_absolute_error, r2_score

from model.train_rent_predictor import (
    load_data,
    clean_data,
    engineer_features,
    prepare_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger: logging.Logger = logging.getLogger(__name__)

# Paths
MODEL_DIR = Path(__file__).parent / "artifacts"
PLOTS_DIR = Path(__file__).parent / "plots"


def setup_plot_style() -> None:
    """Configure matplotlib style for consistent plots."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """
    Scatter plot of predicted vs actual values.
    Perfect predictions would fall on the diagonal line.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors="k", linewidth=0.5)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot(
        [min_val, max_val], [min_val, max_val], "r--", lw=2, label="Predicción perfecta"
    )

    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    ax.set_xlabel("Precio Real (DOP)")
    ax.set_ylabel("Precio Predicho (DOP)")
    ax.set_title(f"Predicción vs Real\nR² = {r2:.4f}, MAE = {mae:,.0f} DOP")
    ax.legend()

    # Format axis with thousands separator
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:,.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:,.0f}"))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
    plt.show()


def plot_residuals_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """
    Histogram of residuals (prediction errors).
    Ideally should be normally distributed around 0.
    """
    residuals = y_pred - y_true

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(x=0, color="r", linestyle="--", linewidth=2, label="Error = 0")
    ax.axvline(
        x=residuals.mean(),
        color="orange",
        linestyle="-",
        linewidth=2,
        label=f"Media = {residuals.mean():,.0f}",
    )

    ax.set_xlabel("Residuos (Predicho - Real) DOP")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"Distribución de Errores\nStd = {residuals.std():,.0f} DOP")
    ax.legend()

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:,.0f}"))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
    plt.show()


def plot_residuals_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """
    Scatter plot of residuals vs predicted values.
    Useful to detect heteroscedasticity (non-constant variance).
    """
    residuals = y_pred - y_true

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(y_pred, residuals, alpha=0.5, edgecolors="k", linewidth=0.5)
    ax.axhline(y=0, color="r", linestyle="--", linewidth=2)

    ax.set_xlabel("Precio Predicho (DOP)")
    ax.set_ylabel("Residuos (DOP)")
    ax.set_title("Residuos vs Valores Predichos")

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:,.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:,.0f}"))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
    plt.show()


def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    top_n: int = 20,
    save_path: Path | None = None,
) -> None:
    """
    Horizontal bar chart of feature importance.
    """
    importance = model.feature_importances_
    indices = np.argsort(importance)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 8))

    cmap = cm.get_cmap("RdYlGn")
    colors = cmap(np.linspace(0.2, 0.8, len(indices)))
    ax.barh(range(len(indices)), importance[indices], color=colors)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])

    ax.set_xlabel("Importancia")
    ax.set_title(f"Top {top_n} Features Más Importantes")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
    plt.show()


def plot_error_by_price_range(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """
    Box plot showing error distribution by price range.
    Helps identify if model performs better/worse in certain ranges.
    """
    # Define price ranges
    bins = [0, 20000, 40000, 60000, 80000, 100000, 150000, 200000, 500000]
    labels = [
        "<20K",
        "20-40K",
        "40-60K",
        "60-80K",
        "80-100K",
        "100-150K",
        "150-200K",
        ">200K",
    ]

    df = pd.DataFrame(
        {
            "actual": y_true,
            "predicted": y_pred,
            "error": y_pred - y_true,
            "abs_error": np.abs(y_pred - y_true),
            "pct_error": np.abs(y_pred - y_true) / y_true * 100,
        }
    )
    df["price_range"] = pd.cut(df["actual"], bins=bins, labels=labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Absolute error by range
    df.boxplot(column="abs_error", by="price_range", ax=axes[0])
    axes[0].set_xlabel("Rango de Precio (DOP)")
    axes[0].set_ylabel("Error Absoluto (DOP)")
    axes[0].set_title("Error Absoluto por Rango de Precio")
    axes[0].tick_params(axis="x", rotation=45)

    # Percentage error by range
    df.boxplot(column="pct_error", by="price_range", ax=axes[1])
    axes[1].set_xlabel("Rango de Precio (DOP)")
    axes[1].set_ylabel("Error Porcentual (%)")
    axes[1].set_title("Error Porcentual por Rango de Precio")
    axes[1].tick_params(axis="x", rotation=45)

    plt.suptitle("")  # Remove automatic title
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
    plt.show()


def plot_accuracy_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tolerance_pct: float = 15.0,
    save_path: Path | None = None,
) -> None:
    """
    Confusion-matrix style visualization for regression.

    Categorizes predictions as:
    - Correct: within tolerance %
    - Overestimated: predicted too high
    - Underestimated: predicted too low

    And shows breakdown by price tier (Low/Medium/High).
    """
    # Define price tiers
    price_33 = np.percentile(y_true, 33)
    price_66 = np.percentile(y_true, 66)

    def get_tier(price: float) -> str:
        if price < price_33:
            return "Bajo"
        elif price < price_66:
            return "Medio"
        return "Alto"

    def get_prediction_category(actual: float, predicted: float) -> str:
        error_pct = (predicted - actual) / actual * 100
        if abs(error_pct) <= tolerance_pct:
            return "Correcto"
        elif error_pct > tolerance_pct:
            return "Sobreestimado"
        return "Subestimado"

    # Classify each prediction
    tiers = [get_tier(y) for y in y_true]
    categories = [get_prediction_category(a, p) for a, p in zip(y_true, y_pred)]

    # Create confusion-style matrix using crosstab
    tier_order = ["Bajo", "Medio", "Alto"]
    category_order = ["Subestimado", "Correcto", "Sobreestimado"]

    # Use crosstab for counting
    matrix = pd.crosstab(
        pd.Categorical(tiers, categories=tier_order),
        pd.Categorical(categories, categories=category_order),
    )
    matrix.index = tier_order
    matrix.columns = category_order

    # Convert to percentages (row-wise)
    matrix_pct = matrix.div(matrix.sum(axis=1), axis=0) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Counts matrix
    axes[0].imshow(matrix.values, cmap="Blues", aspect="auto")
    axes[0].set_xticks(range(len(category_order)))
    axes[0].set_yticks(range(len(tier_order)))
    axes[0].set_xticklabels(category_order)
    axes[0].set_yticklabels(tier_order)
    axes[0].set_xlabel("Categoría de Predicción")
    axes[0].set_ylabel("Rango de Precio Real")
    axes[0].set_title(f"Matriz de Precisión (Tolerancia ±{tolerance_pct}%)\nConteo")

    # Add text annotations
    for i in range(len(tier_order)):
        for j in range(len(category_order)):
            axes[0].text(
                j,
                i,
                f"{matrix.values[i, j]}",
                ha="center",
                va="center",
                fontsize=14,
                color="white"
                if matrix.values[i, j] > matrix.values.max() / 2
                else "black",
            )

    # Percentage matrix
    im2 = axes[1].imshow(
        matrix_pct.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100
    )
    axes[1].set_xticks(range(len(category_order)))
    axes[1].set_yticks(range(len(tier_order)))
    axes[1].set_xticklabels(category_order)
    axes[1].set_yticklabels(tier_order)
    axes[1].set_xlabel("Categoría de Predicción")
    axes[1].set_ylabel("Rango de Precio Real")
    axes[1].set_title(f"Matriz de Precisión (Tolerancia ±{tolerance_pct}%)\nPorcentaje")

    # Add text annotations
    for i in range(len(tier_order)):
        for j in range(len(category_order)):
            axes[1].text(
                j,
                i,
                f"{matrix_pct.values[i, j]:.1f}%",
                ha="center",
                va="center",
                fontsize=12,
                color="white" if matrix_pct.values[i, j] > 50 else "black",
            )

    plt.colorbar(im2, ax=axes[1], label="Porcentaje")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
    plt.show()

    # Print summary
    total_correct = matrix["Correcto"].sum()
    total = matrix.values.sum()
    print(f"\n📊 Resumen de Precisión (±{tolerance_pct}%):")
    print(
        f"   Predicciones correctas: {total_correct}/{total} ({total_correct / total * 100:.1f}%)"
    )
    for tier in tier_order:
        correct_pct = matrix_pct.loc[tier, "Correcto"]
        print(f"   - Precios {tier}s: {correct_pct:.1f}% correctos")


def plot_prediction_error_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """
    Histogram of percentage prediction errors.
    """
    pct_errors = (y_pred - y_true) / y_true * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bins
    bins = np.arange(-100, 105, 5, dtype=float)
    n, bins, patches = ax.hist(
        pct_errors, bins=bins.tolist(), edgecolor="black", alpha=0.7
    )

    # patches is a BarContainer; iterate over its rectangles
    for i, rect in enumerate(patches):
        if i >= len(bins) - 1:
            break
        bin_center = (bins[i] + bins[i + 1]) / 2
        if abs(bin_center) <= 15:
            rect.set_facecolor("#1a9641")  # Green for good predictions
        elif abs(bin_center) <= 30:
            rect.set_facecolor("#fee08b")  # Yellow for okay predictions
        else:
            rect.set_facecolor("#d7191c")  # Red for bad predictions

    ax.axvline(x=0, color="black", linestyle="-", linewidth=2)
    ax.axvline(x=-15, color="green", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(
        x=15,
        color="green",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label="±15% tolerancia",
    )

    ax.set_xlabel("Error Porcentual (%)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de Errores Porcentuales")
    ax.legend()

    within_15 = np.sum(np.abs(pct_errors) <= 15) / len(pct_errors) * 100
    ax.text(
        0.02,
        0.98,
        f"{within_15:.1f}% dentro de ±15%",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
    plt.show()


def plot_learning_curves(
    model: Any,
    X: DataFrame,
    y: pd.Series,
    save_path: Path | None = None,
) -> None:
    """
    Plot learning curves to diagnose bias/variance.
    """
    train_sizes, train_scores, test_scores, *_ = learning_curve(
        model,
        X,
        y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    train_mean = -train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = -test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.1,
        color="blue",
    )
    ax.fill_between(
        train_sizes,
        test_mean - test_std,
        test_mean + test_std,
        alpha=0.1,
        color="orange",
    )

    ax.plot(train_sizes, train_mean, "o-", color="blue", label="Training MAE")
    ax.plot(train_sizes, test_mean, "o-", color="orange", label="Cross-validation MAE")

    ax.set_xlabel("Tamaño del Training Set")
    ax.set_ylabel("MAE (DOP)")
    ax.set_title("Curvas de Aprendizaje")
    ax.legend(loc="best")

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:,.0f}"))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
    plt.show()


def plot_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model: Any,
    feature_names: list[str],
    X: DataFrame | None = None,
    y: pd.Series | None = None,
    save_dir: Path | None = None,
) -> None:
    """
    Generate all visualizations at once.
    """
    setup_plot_style()

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating visualizations...")

    plot_predicted_vs_actual(
        y_true,
        y_pred,
        save_path=save_dir / "01_predicted_vs_actual.png" if save_dir else None,
    )

    plot_residuals_distribution(
        y_true,
        y_pred,
        save_path=save_dir / "02_residuals_distribution.png" if save_dir else None,
    )

    plot_residuals_vs_predicted(
        y_true,
        y_pred,
        save_path=save_dir / "03_residuals_vs_predicted.png" if save_dir else None,
    )

    plot_feature_importance(
        model,
        feature_names,
        save_path=save_dir / "04_feature_importance.png" if save_dir else None,
    )

    plot_error_by_price_range(
        y_true,
        y_pred,
        save_path=save_dir / "05_error_by_price_range.png" if save_dir else None,
    )

    plot_accuracy_matrix(
        y_true,
        y_pred,
        save_path=save_dir / "06_accuracy_matrix.png" if save_dir else None,
    )

    plot_prediction_error_histogram(
        y_true,
        y_pred,
        save_path=save_dir / "07_error_histogram.png" if save_dir else None,
    )

    if X is not None and y is not None:
        plot_learning_curves(
            model,
            X,
            y,
            save_path=save_dir / "08_learning_curves.png" if save_dir else None,
        )

    logger.info("All visualizations generated!")


def main(input_file: str, save_plots: bool = True) -> None:
    """
    Main function to generate all visualizations.
    """
    setup_plot_style()

    # Load model and encoders
    model = joblib.load(MODEL_DIR / "rent_predictor.joblib")
    encoders = joblib.load(MODEL_DIR / "encoders.joblib")

    # Load and process data (same as training)
    df = load_data(input_file)
    df = clean_data(df)
    df = engineer_features(df)

    # Prepare features
    X, _ = prepare_features(df, encoders, fit_encoders=False)
    y = df["rent"]

    # Split data (same random state as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Get predictions
    y_pred = model.predict(X_test)

    # Generate all plots
    save_dir = PLOTS_DIR if save_plots else None

    plot_all(
        y_true=y_test.values,
        y_pred=y_pred,
        model=model,
        feature_names=list(X.columns),
        X=X,
        y=y,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate visualizations for rent prediction model"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/supercasas_apartments.csv",
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save plots to files (only display)",
    )

    args = parser.parse_args()
    main(input_file=args.input, save_plots=not args.no_save)
