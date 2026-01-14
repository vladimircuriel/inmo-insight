"""
Load module for Supercasas.com data.

This module is responsible for:
- Loading enriched data to PostgreSQL database
- Supporting both in-memory data and CSV file loading
- Creating/updating database schema as needed

Input: Enriched data from enrich.supercasas or CSV file
Output: Data persisted in PostgreSQL
"""

import os
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# Load environment variables from .env file (in airflow directory)
AIRFLOW_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=AIRFLOW_DIR / ".env")

logger: logging.Logger = logging.getLogger(name=__name__)

# Database configuration from environment
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# Table name
TABLE_NAME = "supercasas_apartments"

# Default CSV path (from enrich module)
DEFAULT_CSV_PATH = "data/supercasas_enriched.csv"


def get_database_url() -> str:
    """Build PostgreSQL connection URL from environment variables."""
    if not all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD]):
        raise ValueError(
            "Missing database credentials. Check .env file for: "
            "DB_HOST, DB_NAME, DB_USER, DB_PASSWORD"
        )
    return f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def get_engine() -> Engine:
    """Create SQLAlchemy engine for PostgreSQL connection."""
    return create_engine(get_database_url())


def create_table_if_not_exists(engine: Engine) -> None:
    """Create the apartments table if it doesn't exist."""
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        id SERIAL PRIMARY KEY,
        url TEXT UNIQUE,
        site_id VARCHAR(50) UNIQUE,
        
        -- Price info
        original_rent NUMERIC,
        rent NUMERIC,
        rent_with_utilities NUMERIC,
        origin_currency VARCHAR(10),
        
        -- Basic info
        rooms INTEGER,
        baths NUMERIC,
        parking INTEGER,
        
        -- Location
        city VARCHAR(100),
        location VARCHAR(255),
        latitude NUMERIC,
        longitude NUMERIC,
        city_validated BOOLEAN,
        
        -- Property details
        previous_owners VARCHAR(100),
        construction_meters NUMERIC,
        floor VARCHAR(50),
        is_modificable VARCHAR(50),
        use_type VARCHAR(100),
        land_meters NUMERIC,
        amount_of_elevators INTEGER,
        year_built VARCHAR(20),
        
        -- Facilities
        facilities TEXT[],
        amount_of_facilities INTEGER,
        observations TEXT,
        
        -- Enrichment: Conflict detection
        city_conflict BOOLEAN,
        location_conflict BOOLEAN,
        construction_meters_conflict BOOLEAN,
        elevators_conflict BOOLEAN,
        rent_mentions_conflict BOOLEAN,
        
        -- Enrichment: Text extraction
        construction_meters_text TEXT,
        elevators_text TEXT,
        rent_text TEXT,
        
        -- Enrichment: Room features
        service_room BOOLEAN,
        service_bath BOOLEAN,
        walk_in_closet BOOLEAN,
        
        -- Enrichment: Amenities
        has_pool BOOLEAN,
        has_gym BOOLEAN,
        has_terrace BOOLEAN,
        has_bbq_area BOOLEAN,
        has_kids_area BOOLEAN,
        has_multiuse_room BOOLEAN,
        has_gazebo BOOLEAN,
        
        -- Enrichment: Building features
        full_power_plant BOOLEAN,
        water_cistern BOOLEAN,
        water_well BOOLEAN,
        common_gas BOOLEAN,
        security_cameras BOOLEAN,
        electric_gate BOOLEAN,
        security_24_7 BOOLEAN,
        
        -- Enrichment: Commercial info
        negotiable BOOLEAN,
        maintenance_mentioned BOOLEAN,
        
        -- Enrichment: Contact info
        has_contact_phone BOOLEAN,
        phone_text TEXT,
        agent_name_text TEXT,
        
        -- Metadata
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Create index on site_id for faster lookups
    CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_site_id ON {TABLE_NAME}(site_id);
    """

    with engine.begin() as conn:
        conn.execute(text(create_table_sql))

    logger.info(f"Table '{TABLE_NAME}' is ready")


def prepare_dataframe_for_db(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame for database insertion."""
    df = df.copy()

    # Convert facilities list to PostgreSQL array format
    if "facilities" in df.columns:
        df["facilities"] = df["facilities"].apply(
            lambda x: x if isinstance(x, list) else []
        )

    # Handle NaN values - replace with None for proper SQL NULL
    df = df.replace({pd.NA: None, float("nan"): None})

    return df


def load_to_database(
    data: list[dict[str, Any]] | pd.DataFrame,
    if_exists: str = "replace",
) -> int:
    """
    Load enriched apartment data to PostgreSQL.

    Args:
        data: List of apartment dictionaries or DataFrame
        if_exists: How to behave if table exists ('replace', 'append', 'fail')

    Returns:
        Number of rows inserted
    """
    logger.info("Connecting to database...")
    engine = get_engine()

    # Create table if needed
    create_table_if_not_exists(engine)

    # Convert to DataFrame if needed
    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data

    if df.empty:
        logger.warning("No data to load")
        return 0

    # Prepare data
    df = prepare_dataframe_for_db(df)

    # Load to database
    logger.info(f"Loading {len(df)} rows to '{TABLE_NAME}'...")

    # For 'replace' mode, we drop and recreate to handle schema changes
    if if_exists == "replace":
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {TABLE_NAME}"))
        create_table_if_not_exists(engine)

    # Use pandas to_sql for simple insertion
    df.to_sql(
        name=TABLE_NAME,
        con=engine,
        if_exists="append",
        index=False,
        method="multi",
    )

    logger.info(f"Successfully loaded {len(df)} rows")
    return len(df)


def load_from_csv(
    csv_path: str = DEFAULT_CSV_PATH,
    if_exists: str = "replace",
) -> int:
    """
    Load data from CSV file to PostgreSQL.

    Args:
        csv_path: Path to the CSV file
        if_exists: How to behave if table exists ('replace', 'append', 'fail')

    Returns:
        Number of rows inserted
    """
    logger.info(f"Loading data from CSV: {csv_path}")

    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Parse facilities column if it's a string representation of list
    if "facilities" in df.columns:
        import ast

        df["facilities"] = df["facilities"].apply(
            lambda x: ast.literal_eval(x)
            if isinstance(x, str) and x.startswith("[")
            else []
        )

    logger.info(f"Loaded {len(df)} rows from CSV")

    return load_to_database(data=df, if_exists=if_exists)


def load_supercasas_data(
    enriched_data: list[dict[str, Any]],
    if_exists: str = "replace",
) -> int:
    """
    Load enriched apartment data to PostgreSQL (main entry point).

    Args:
        enriched_data: List of enriched apartment dictionaries
        if_exists: How to behave if table exists ('replace', 'append', 'fail')

    Returns:
        Number of rows inserted
    """
    logger.info("STARTING LOAD PROCESS")
    rows = load_to_database(data=enriched_data, if_exists=if_exists)
    logger.info("LOAD PROCESS COMPLETED")
    return rows


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Load supercasas data to PostgreSQL")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to CSV file to load. If not provided, runs full pipeline.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["replace", "append"],
        default="replace",
        help="How to handle existing data (default: replace)",
    )

    args = parser.parse_args()

    if args.csv:
        # Load from CSV file
        rows = load_from_csv(csv_path=args.csv, if_exists=args.mode)
        print(f"Loaded {rows} rows from CSV to database")
    else:
        # Run full pipeline: extract -> transform -> enrich -> load
        from extract.supercasas import extract_from_supercasas
        from transform.supercasas import transform_supercasas_data
        from enrich.supercasas import enrich_supercasas_data

        # Test limit
        TEST_LIMIT = 5

        raw_data = extract_from_supercasas(max_apartments=TEST_LIMIT)
        transformed_data = transform_supercasas_data(
            raw_data=raw_data, save_to_csv=False
        )
        enriched_data = enrich_supercasas_data(
            transformed_data=transformed_data,
            wait_for_completion=True,
        )

        if isinstance(enriched_data, list):
            rows = load_supercasas_data(
                enriched_data=enriched_data, if_exists=args.mode
            )
            print(f"Loaded {rows} rows to database")
