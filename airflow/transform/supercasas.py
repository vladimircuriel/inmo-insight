"""
Transform module for Supercasas.com data.

This module is responsible for:
- Data cleaning and normalization
- Type conversions (strings to numeric)
- Currency conversions (USD to DOP)
- Business logic and derived fields
- Null value handling

Input: Raw extracted data from extract.supercasas
Output: Clean, transformed data ready for loading
"""

import re
import logging
import pandas as pd
from pathlib import Path
from typing import Any

from common.utils import get_usd_to_dop_rate

logger: logging.Logger = logging.getLogger(name=__name__)

# Fetch current exchange rate at module load
USD_TO_DOP_RATE = get_usd_to_dop_rate()

# Output file path
CSV_FILE_NAME = "data/supercasas_data.csv"

# Fields where 0 semantically means "not informed"
ZERO_AS_NULL_FIELDS: set[str] = {
    "construction_meters",
    "land_meters",
}

# String values that represent null/missing data
STRING_NULL_VALUES: set[str] = {"n/d", "no informado", ""}

# Regex patterns for currency detection
USD_PATTERNS = [
    r"US\$\s*([\d,]+(?:\.\d{2})?)",  # US$ 1,200 or US$ 1,200.00
    r"USD\s*([\d,]+(?:\.\d{2})?)",  # USD 1200
    r"\$US\s*([\d,]+(?:\.\d{2})?)",  # $US 1200
    r"\$USD\s*([\d,]+(?:\.\d{2})?)",  # $USD 1200
    r"U\$D?\s*([\d,]+(?:\.\d{2})?)",  # U$ 1200 or U$D 1200
    r"(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:USD|US\$|dolares?|dólares?)",  # 1,200 USD
]

DOP_PATTERNS = [
    r"RD\$\s*([\d,]+(?:\.\d{2})?)",  # RD$ 25,000
    r"\$RD\s*([\d,]+(?:\.\d{2})?)",  # $RD 25,000
    r"DOP\s*([\d,]+(?:\.\d{2})?)",  # DOP 25000
    r"(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:RD\$|DOP|pesos?)",  # 25,000 RD$
]


def detect_currency_and_amount(raw_value: str) -> tuple[str | None, float | None]:
    """
    Detect currency and extract amount from raw price string.

    Args:
        raw_value: Raw price string

    Returns:
        Tuple of (currency, amount) or (None, None) if not found
    """
    if not raw_value:
        return None, None

    # Normalize the string
    text = raw_value.upper()

    # Check USD patterns first (more common for real estate)
    for pattern in USD_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            amount_str = match.group(1).replace(",", "")
            return "USD", float(amount_str)

    # Check DOP patterns
    for pattern in DOP_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            amount_str = match.group(1).replace(",", "")
            return "DOP", float(amount_str)

    # Fallback: try to find a standalone number (assume USD if > 10000 threshold suggests DOP)
    # Numbers like 1200, 1,500 without currency marker
    match = re.search(r"(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)", raw_value)
    if match:
        amount_str = match.group(1).replace(",", "")
        amount = float(amount_str)
        # Heuristic: amounts > 10,000 are likely DOP, smaller amounts likely USD
        # Based on typical rent prices in DR
        if amount > 10000:
            return "DOP", amount
        else:
            return "USD", amount

    return None, None


def parse_price(raw_value: str | None) -> dict[str, Any]:
    """
    Parse raw price string and extract amount and currency.

    Args:
        raw_value: Raw price string like "Alquiler: RD$ 25,000" or "US$ 500"

    Returns:
        Dictionary with 'amount', 'currency', and 'amount_dop' keys
    """
    result: dict[str, Any] = {
        "amount": None,
        "currency": None,
        "amount_dop": None,
    }

    if not raw_value:
        return result

    currency, amount = detect_currency_and_amount(raw_value)

    if currency is None or amount is None:
        return result

    amount_dop = amount if currency == "DOP" else amount * USD_TO_DOP_RATE

    result["amount"] = amount
    result["currency"] = currency
    result["amount_dop"] = amount_dop

    return result


def parse_numeric(raw_value: str | None, as_int: bool = False) -> int | float | None:
    """
    Extract numeric value from raw string.

    Args:
        raw_value: Raw string potentially containing a number
        as_int: If True, return integer; otherwise return float

    Returns:
        Parsed numeric value or None
    """
    if not raw_value:
        return None

    match = re.search(pattern=r"(\d+(?:\.\d+)?)", string=raw_value)
    if not match:
        return None

    value = float(match.group(1))
    return int(value) if as_int else value


def parse_location(raw_value: str | None) -> dict[str, str | None]:
    """
    Parse location string to extract city and specific location.

    Args:
        raw_value: Raw location string like "Santo Domingo > Piantini"

    Returns:
        Dictionary with 'city' and 'location' keys
    """
    result: dict[str, str | None] = {"city": None, "location": None}

    if not raw_value:
        return result

    if ">" in raw_value:
        parts = raw_value.split(sep=">", maxsplit=1)
        result["city"] = parts[0].strip()
        result["location"] = parts[1].strip() if len(parts) > 1 else None
    else:
        result["location"] = raw_value.strip()

    return result


def normalize_null_values(value: Any, field_name: str) -> Any:
    """
    Normalize placeholder values to None.

    - 0 / 0.0 for selected numeric fields where zero means 'not informed'
    - 'N/D', 'No informado', empty strings for string fields

    Args:
        value: The value to normalize
        field_name: The name of the field (for context-specific rules)

    Returns:
        Normalized value (or None if it represents missing data)
    """
    # Handle numeric fields where 0 means "not informed"
    if field_name in ZERO_AS_NULL_FIELDS:
        if isinstance(value, (int, float)) and value == 0:
            return None

    # Handle string null placeholders
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.lower() in STRING_NULL_VALUES:
            return None

    return value


def transform_apartment(raw_data: dict[str, Any]) -> dict[str, Any]:
    """
    Transform a single raw apartment record.

    Applies all transformations:
    - Parse and convert prices (USD → DOP)
    - Convert numeric strings to proper types
    - Parse location into city/location
    - Calculate derived fields
    - Normalize null values

    Args:
        raw_data: Raw apartment dictionary from extraction

    Returns:
        Transformed apartment dictionary
    """
    transformed: dict[str, Any] = {
        "url": raw_data.get("url"),
        "site_id": raw_data.get("site_id"),
    }

    # Parse rent prices
    rent_info = parse_price(raw_data.get("rent"))
    transformed["original_rent"] = rent_info["amount"]
    transformed["rent"] = rent_info["amount_dop"]
    transformed["origin_currency"] = rent_info["currency"]

    # Parse rent with utilities
    rent_utils_info = parse_price(raw_data.get("rent_with_utilities"))
    transformed["rent_with_utilities"] = rent_utils_info["amount_dop"]

    # If origin_currency is missing but rent_with_utilities has currency, use that
    if transformed["origin_currency"] is None and rent_utils_info["currency"]:
        transformed["origin_currency"] = rent_utils_info["currency"]

    # Parse numeric fields
    transformed["rooms"] = parse_numeric(raw_data.get("rooms"), as_int=True)
    transformed["baths"] = parse_numeric(raw_data.get("baths"), as_int=False)
    transformed["parking"] = parse_numeric(raw_data.get("parking"), as_int=True)

    # Parse location
    location_info = parse_location(raw_data.get("location"))
    transformed["city"] = location_info["city"]
    transformed["location"] = location_info["location"]

    # Parse general info numeric fields
    transformed["construction_meters"] = parse_numeric(
        raw_data.get("construction_meters")
    )
    transformed["land_meters"] = parse_numeric(raw_data.get("land_meters"))
    transformed["amount_of_elevators"] = parse_numeric(
        raw_data.get("amount_of_elevators"), as_int=True
    )

    # Pass through string fields (with normalization)
    string_fields = [
        "previous_owners",
        "floor",
        "is_modificable",
        "use_type",
        "year_built",
        "observations",
    ]
    for field in string_fields:
        transformed[field] = raw_data.get(field)

    # Handle facilities list and calculate derived field
    facilities = raw_data.get("facilities", [])
    if not isinstance(facilities, list):
        facilities = []
    transformed["facilities"] = facilities
    transformed["amount_of_facilities"] = len(facilities)

    # Normalize null values for all fields
    for key in list(transformed.keys()):
        transformed[key] = normalize_null_values(transformed[key], key)

    return transformed


def transform_supercasas_data(
    raw_data: list[dict[str, Any]],
    save_to_csv: bool = True,
) -> list[dict[str, Any]]:
    """
    Transform all raw apartment data from supercasas.

    Args:
        raw_data: List of raw apartment dictionaries from extraction
        save_to_csv: Whether to save transformed data to CSV

    Returns:
        List of transformed apartment dictionaries
    """
    logger.info(msg="STARTING TRANSFORMATION PROCESS")

    transformed_data: list[dict[str, Any]] = []

    for raw_apt in raw_data:
        try:
            transformed_apt = transform_apartment(raw_data=raw_apt)
            transformed_data.append(transformed_apt)
        except Exception as e:
            logger.warning(
                msg=f"Failed to transform apartment {raw_apt.get('url')}: {e}"
            )
            continue

    logger.info(msg=f"TRANSFORMED {len(transformed_data)} APARTMENTS")

    if save_to_csv and transformed_data:
        df = pd.DataFrame(data=transformed_data)
        logger.info(msg=f"DATAFRAME SHAPE: {df.shape}")

        Path(CSV_FILE_NAME).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path_or_buf=CSV_FILE_NAME, index=False)
        logger.info(msg=f"SAVED TO {CSV_FILE_NAME}")

    logger.info(msg="TRANSFORMATION PROCESS COMPLETED")
    return transformed_data


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger: logging.Logger = logging.getLogger(name=__name__)

    # Example: load raw data and transform
    from extract.supercasas import extract_from_supercasas

    raw_data = extract_from_supercasas()
    transformed_data = transform_supercasas_data(raw_data=raw_data)
    print(f"Transformed {len(transformed_data)} apartments")

    # export to CSV
    df = pd.DataFrame(data=transformed_data)
    df.to_csv("supercasas_transformed_data.csv", index=False)
