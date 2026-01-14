"""
Apartment Rent Predictor - XGBoost Regressor

This script trains a model to predict apartment rent prices based on
features extracted from supercasas.com listings.

Usage:
    python -m model.train_rent_predictor --input data/supercasas_apartments.csv
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pandas import DataFrame

# Opt-in to future pandas behavior to avoid FutureWarning
pd.set_option("future.no_silent_downcasting", True)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Output paths
MODEL_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH = MODEL_DIR / "rent_predictor.joblib"
ENCODERS_PATH = MODEL_DIR / "encoders.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"


def load_data(filepath: str) -> DataFrame:
    """Load and return the raw dataset."""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows")
    return df


def clean_data(df: DataFrame) -> DataFrame:
    """
    Clean the dataset:
    - Remove rows without rent price
    - Remove outliers
    - Handle missing values
    """
    logger.info("Cleaning data...")

    # Keep only rows with valid rent (target variable)
    df = df[df["rent"].notna() & (df["rent"] > 0)].copy()
    logger.info(f"Rows with valid rent: {len(df)}")

    # Remove extreme outliers (rent > 500,000 DOP or < 5,000 DOP)
    df = df[(df["rent"] >= 5000) & (df["rent"] <= 500000)]
    logger.info(f"Rows after removing outliers: {len(df)}")

    # Fill missing numeric values with median
    numeric_cols = [
        "rooms",
        "baths",
        "parking",
        "construction_meters",
        "floor",
        "amount_of_elevators",
        "amount_of_facilities",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Fill missing categorical values
    categorical_cols = [
        "city",
        "location",
        "origin_currency",
        "previous_owners",
        "is_modificable",
        "use_type",
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Fill boolean columns with False
    bool_cols = [
        "has_pool",
        "has_gym",
        "has_terrace",
        "has_bbq_area",
        "has_kids_area",
        "has_multiuse_room",
        "has_gazebo",
        "full_power_plant",
        "water_cistern",
        "water_well",
        "common_gas",
        "security_cameras",
        "electric_gate",
        "security_24_7",
        "service_room",
        "service_bath",
        "walk_in_closet",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)

    return df


def engineer_features(df: DataFrame) -> DataFrame:
    """
    Create new features from existing data.
    """
    logger.info("Engineering features...")

    # Price per square meter (if construction_meters available)
    if "construction_meters" in df.columns:
        df["price_per_sqm"] = df.apply(
            lambda row: row["rent"] / row["construction_meters"]
            if pd.notna(row["construction_meters"]) and row["construction_meters"] > 0
            else np.nan,
            axis=1,
        )

    # Total rooms (rooms + baths)
    if "rooms" in df.columns and "baths" in df.columns:
        df["total_rooms"] = df["rooms"].fillna(0) + df["baths"].fillna(0)

    # Has parking
    if "parking" in df.columns:
        df["has_parking"] = df["parking"].fillna(0) > 0

    # Is high floor
    if "floor" in df.columns:
        df["is_high_floor"] = df["floor"].fillna(0) >= 5

    # Amenities score (count of boolean amenities)
    bool_cols = [
        "has_pool",
        "has_gym",
        "has_terrace",
        "has_bbq_area",
        "has_kids_area",
        "has_multiuse_room",
        "has_gazebo",
        "full_power_plant",
        "water_cistern",
        "security_24_7",
        "service_room",
        "walk_in_closet",
    ]
    existing_bool_cols = [col for col in bool_cols if col in df.columns]
    if existing_bool_cols:
        df["amenities_score"] = df[existing_bool_cols].sum(axis=1)

    # Is new property
    if "previous_owners" in df.columns:
        df["is_new"] = (
            df["previous_owners"].str.lower().str.contains("nueva|nuevo", na=False)
        )

    return df


def prepare_features(
    df: DataFrame,
    encoders: dict[str, LabelEncoder] | None = None,
    fit_encoders: bool = True,
) -> tuple[DataFrame, dict[str, LabelEncoder]]:
    """
    Prepare features for training:
    - Label encode categorical variables
    - Select relevant features
    """
    logger.info("Preparing features...")

    if encoders is None:
        encoders = {}

    # Features to use
    numeric_features = [
        "rooms",
        "baths",
        "parking",
        "construction_meters",
        "floor",
        "amount_of_elevators",
        "amount_of_facilities",
        "amenities_score",
    ]

    # origin_currency indicates market segment (USD = premium properties)
    categorical_features = ["location", "origin_currency"]

    boolean_features = [
        # Amenities
        "has_pool",
        "has_gym",
        "has_terrace",
        "has_bbq_area",
        "has_kids_area",
        "has_multiuse_room",
        "has_gazebo",
        # Infrastructure
        "full_power_plant",
        "water_cistern",
        "water_well",
        "common_gas",
        # Security
        "security_cameras",
        "electric_gate",
        "security_24_7",
        # Interior features
        "service_room",
        "service_bath",
        "walk_in_closet",
        # Other
        "negotiable",
        "maintenance_mentioned",
        "has_contact_phone",
        # Engineered
        "is_high_floor",
        "is_new",
    ]

    # Filter to existing columns
    numeric_features = [f for f in numeric_features if f in df.columns]
    categorical_features = [f for f in categorical_features if f in df.columns]
    boolean_features = [f for f in boolean_features if f in df.columns]

    # Create feature dataframe
    X = df[numeric_features].copy()

    # Label encode categorical features
    for col in categorical_features:
        col_values = df[col].fillna("Unknown").astype(str)

        if fit_encoders:
            if col not in encoders:
                encoders[col] = LabelEncoder()
            encoders[col].fit(col_values)

        X[col] = encoders[col].transform(col_values)

    # Add boolean features (convert to int)
    for col in boolean_features:
        X[col] = df[col].fillna(False).astype(int)

    # Fill any remaining NaN with 0
    X = X.fillna(0)

    logger.info(f"Prepared {len(X.columns)} features: {list(X.columns)}")

    return X, encoders


def train_model(
    X_train: DataFrame, y_train: pd.Series, use_grid_search: bool = True
) -> xgb.XGBRegressor:
    """
    Train XGBoost regressor with optional grid search.
    """
    logger.info("Training model...")

    if use_grid_search:
        logger.info("Performing grid search for hyperparameter tuning...")

        base_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )

        param_grid = {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 1.0],
        }

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="neg_mean_absolute_error",
            cv=3,
            verbose=1,
            n_jobs=-1,
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score (MAE): {-grid_search.best_score_:.2f}")

        model = grid_search.best_estimator_
    else:
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

    return model


def evaluate_model(
    model: xgb.XGBRegressor, X_test: DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """
    Evaluate model performance.
    """
    logger.info("Evaluating model...")

    y_pred = model.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
        "mape": np.mean(np.abs((y_test - y_pred) / y_test)) * 100,
    }

    logger.info("Model Performance:")
    logger.info(f"  MAE:  {metrics['mae']:,.2f} DOP")
    logger.info(f"  RMSE: {metrics['rmse']:,.2f} DOP")
    logger.info(f"  R²:   {metrics['r2']:.4f}")
    logger.info(f"  MAPE: {metrics['mape']:.2f}%")

    return metrics


def get_feature_importance(
    model: xgb.XGBRegressor, feature_names: list[str]
) -> DataFrame:
    """
    Get feature importance from the trained model.
    """
    importance = model.feature_importances_

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importance}
    ).sort_values("importance", ascending=False)

    logger.info("\nTop 10 Feature Importances:")
    for _, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    return importance_df


def save_artifacts(
    model: xgb.XGBRegressor,
    encoders: dict[str, LabelEncoder],
    feature_names: list[str],
    metrics: dict[str, float],
    best_params: dict[str, Any] | None = None,
) -> None:
    """
    Save model and preprocessing artifacts.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save model
    joblib.dump(model, MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")

    # Save encoders
    joblib.dump(encoders, ENCODERS_PATH)
    logger.info(f"Encoders saved to {ENCODERS_PATH}")

    # Save metadata
    metadata = {
        "feature_names": feature_names,
        "metrics": metrics,
        "best_params": best_params or model.get_params(),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {METADATA_PATH}")


def main(input_path: str, use_grid_search: bool = False) -> None:
    """
    Main training pipeline.
    """
    # Load data
    df = load_data(input_path)

    # Clean data
    df = clean_data(df)

    # Engineer features
    df = engineer_features(df)

    # Prepare features
    X, encoders = prepare_features(df, fit_encoders=True)
    y = df["rent"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Train model
    model = train_model(X_train, y_train, use_grid_search=use_grid_search)

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    # Feature importance
    get_feature_importance(model, list(X.columns))

    # Save artifacts
    save_artifacts(
        model=model,
        encoders=encoders,
        feature_names=list(X.columns),
        metrics=metrics,
    )

    # Save predictions for analysis
    predictions_df = X_test.copy()
    predictions_df["actual_rent"] = y_test.values
    predictions_df["predicted_rent"] = model.predict(X_test)
    predictions_df["error"] = (
        predictions_df["predicted_rent"] - predictions_df["actual_rent"]
    )
    predictions_df["error_pct"] = (
        predictions_df["error"] / predictions_df["actual_rent"]
    ) * 100

    predictions_path = MODEL_DIR / "test_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f"Test predictions saved to {predictions_path}")

    logger.info("\n Training complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train apartment rent predictor")
    parser.add_argument(
        "--input",
        type=str,
        default="data/supercasas_apartments.csv",
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Use grid search for hyperparameter tuning (slower but better)",
    )

    args = parser.parse_args()
    main(args.input, args.grid_search)
