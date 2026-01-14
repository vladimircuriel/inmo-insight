from datetime import datetime, timedelta
from typing import Any

from airflow.sdk import dag, task

from extract.supercasas import extract_from_supercasas
from transform.supercasas import transform_supercasas_data
from enrich.supercasas import enrich_supercasas_data
from load.supercasas import load_supercasas_data


@dag(
    start_date=datetime(year=2026, month=1, day=1),
    schedule=timedelta(weeks=2),
    catchup=False,
    tags=["supercasas", "real-estate"],
)
def inmo_insight_pipeline():
    """
    Pipeline ETL para datos de apartamentos de Supercasas.com

    Stages:
    1. Extract: Scraping de datos crudos desde supercasas.com
    2. Transform: Limpieza, normalización y conversión de tipos
    3. Enrich: Enriquecimiento con OpenAI Batch API
    4. Load: Carga a PostgreSQL
    """

    @task
    def extract_data(max_apartments: int | None = None) -> list[dict[str, Any]]:
        """Extract raw apartment data from supercasas.com"""
        return extract_from_supercasas(max_apartments=max_apartments)

    @task
    def transform_data(raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Transform and normalize raw data"""
        return transform_supercasas_data(raw_data=raw_data, save_to_csv=False)

    @task
    def enrich_data(transformed_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Enrich data using OpenAI Batch API"""
        result = enrich_supercasas_data(
            transformed_data=transformed_data,
            wait_for_completion=True,
        )
        # enrich_supercasas_data can return batch_id (str) if wait_for_completion=False
        if isinstance(result, str):
            raise ValueError(f"Expected enriched data, got batch_id: {result}")
        return result

    @task
    def load_data(enriched_data: list[dict[str, Any]]) -> int:
        """Load enriched data to PostgreSQL"""
        return load_supercasas_data(enriched_data=enriched_data, if_exists="replace")

    # Pipeline flow
    raw_data = extract_data()
    transformed_data = transform_data(raw_data=raw_data)
    enriched_data = enrich_data(transformed_data=transformed_data)
    load_data(enriched_data=enriched_data)


inmo_insight_pipeline()
