"""
Prediction module for apartment rent prices.

Usage:
    from model.predict import RentPredictor

    predictor = RentPredictor()
    price = predictor.predict({
        "rooms": 3,
        "baths": 2,
        "location": "Bella Vista",
        ...
    })
"""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from pandas import DataFrame

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "artifacts"


class RentPredictor:
    """
    Predictor class for apartment rent prices.
    """

    def __init__(self, model_dir: Path | str = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.model: Any = None
        self.encoders: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load model and preprocessing artifacts."""
        model_path = self.model_dir / "rent_predictor.joblib"
        encoders_path = self.model_dir / "encoders.joblib"
        metadata_path = self.model_dir / "metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run train_rent_predictor.py first."
            )

        self.model = joblib.load(filename=model_path)
        self.encoders = joblib.load(filename=encoders_path)

        with open(metadata_path) as f:
            self.metadata = json.load(f)

        logger.info("Model artifacts loaded successfully")

    def _prepare_input(self, data: dict[str, Any]) -> DataFrame:
        """
        Prepare input data for prediction.
        """
        feature_names = self.metadata["feature_names"]

        # Create DataFrame with single row
        df = pd.DataFrame([data])

        # Engineer features
        if "rooms" in df.columns and "baths" in df.columns:
            df["total_rooms"] = df["rooms"].fillna(0) + df["baths"].fillna(0)

        if "parking" in df.columns:
            df["has_parking"] = df["parking"].fillna(0) > 0

        if "floor" in df.columns:
            df["is_high_floor"] = df["floor"].fillna(0) >= 5

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

        if "previous_owners" in df.columns:
            df["is_new"] = (
                df["previous_owners"].str.lower().str.contains("nueva|nuevo", na=False)
            )

        # Prepare feature vector
        X = pd.DataFrame(columns=feature_names)

        for col in feature_names:
            if col in df.columns:
                if col in self.encoders:
                    # Label encoded categorical feature
                    value = (
                        str(df[col].iloc[0]) if pd.notna(df[col].iloc[0]) else "Unknown"
                    )
                    try:
                        X[col] = self.encoders[col].transform([value])
                    except ValueError:
                        # Unknown category - use 0
                        X[col] = [0]
                else:
                    # Numeric or boolean feature
                    X[col] = df[col].fillna(0).astype(float)
            else:
                X[col] = [0]

        return X

    def predict(self, data: dict[str, Any]) -> float:
        """
        Predict rent price for a single apartment.

        Args:
            data: Dictionary with apartment features

        Returns:
            Predicted rent price in DOP
        """
        X = self._prepare_input(data)
        prediction = self.model.predict(X)[0]
        return float(prediction)

    def predict_batch(self, data: list[dict[str, Any]]) -> list[float]:
        """
        Predict rent prices for multiple apartments.
        """
        return [self.predict(d) for d in data]

    def get_model_info(self) -> dict[str, Any]:
        """
        Get model information and metrics.
        """
        return {
            "feature_names": self.metadata["feature_names"],
            "metrics": self.metadata["metrics"],
            "n_features": len(self.metadata["feature_names"]),
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example prediction
    predictor = RentPredictor()

    sample_apartment = {
        "rooms": 3,
        "baths": 2,
        "parking": 2,
        "construction_meters": 120,
        "floor": 5,
        "city": "Santiago",
        "location": "Bella Vista",
        "origin_currency": "USD",
        "use_type": "Residencial",
        "has_pool": True,
        "has_gym": True,
        "security_24_7": True,
        "amount_of_facilities": 10,
        "amount_of_elevators": 2,
    }

    predicted_rent = predictor.predict(sample_apartment)
    print(f"\nPredicted rent: RD${predicted_rent:,.2f}")

    # Model info
    info = predictor.get_model_info()
    print("\nModel metrics:")
    print(f"  MAE:  {info['metrics']['mae']:,.2f} DOP")
    print(f"  R²:   {info['metrics']['r2']:.4f}")
