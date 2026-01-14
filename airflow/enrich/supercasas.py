"""
Enrich module for Supercasas.com data using OpenAI Batch API.

This module is responsible for:
- Preparing transformed data for OpenAI Batch API
- Creating JSONL batch input files
- Submitting batches to OpenAI
- Processing batch results and merging enrichments

Input: Transformed data from transform.supercasas
Output: Enriched data with AI-extracted fields
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from common.constants import OPENAI_ENRICHMENT_SYSTEM_PROMPT
from common.santiago_geolocation import get_santiago_coordinates

logger: logging.Logger = logging.getLogger(name=__name__)

# Load environment variables from .env file (in airflow directory)
AIRFLOW_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=AIRFLOW_DIR / ".env")

# File paths
BATCH_INPUT_FILE = "data/batch_input.jsonl"
BATCH_OUTPUT_FILE = "data/batch_output.jsonl"
ENRICHED_CSV_FILE = "data/supercasas_enriched.csv"

# OpenAI Batch API settings
OPENAI_MODEL = "gpt-5-nano"
BATCH_ENDPOINT = "/v1/chat/completions"

# Enrichment fields that will be added by OpenAI
ENRICHMENT_FIELDS: list[str] = [
    "city_conflict",
    "location_conflict",
    "construction_meters_conflict",
    "elevators_conflict",
    "rent_mentions_conflict",
    "construction_meters_text",
    "elevators_text",
    "rent_text",
    "service_room",
    "service_bath",
    "walk_in_closet",
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
    "negotiable",
    "maintenance_mentioned",
    "has_contact_phone",
    "phone_text",
    "agent_name_text",
]


def create_batch_request(
    row: dict[str, Any],
    custom_id: str,
) -> dict[str, Any]:
    """
    Create a single batch request for the OpenAI Batch API.

    Args:
        row: Transformed apartment data dictionary
        custom_id: Unique identifier for this request (e.g., site_id)

    Returns:
        Batch request dictionary in OpenAI format
    """
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": BATCH_ENDPOINT,
        "body": {
            "model": OPENAI_MODEL,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "developer",
                    "content": OPENAI_ENRICHMENT_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": f"CSV Row:\n\n{json.dumps(row, indent=2, ensure_ascii=False)}",
                },
            ],
        },
    }


def prepare_batch_input(
    transformed_data: list[dict[str, Any]],
    output_path: str = BATCH_INPUT_FILE,
) -> str:
    """
    Prepare JSONL file for OpenAI Batch API.

    Args:
        transformed_data: List of transformed apartment dictionaries
        output_path: Path to write the JSONL file

    Returns:
        Path to the created JSONL file
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for row in transformed_data:
            site_id = row.get("site_id", "unknown")
            custom_id = f"apt-{site_id}"

            batch_request = create_batch_request(row=row, custom_id=custom_id)
            f.write(json.dumps(batch_request, ensure_ascii=False) + "\n")

    return output_path


def submit_batch(
    client: OpenAI,
    input_file_path: str = BATCH_INPUT_FILE,
) -> str:
    """
    Submit a batch job to OpenAI Batch API.

    Args:
        client: OpenAI client instance
        input_file_path: Path to the JSONL input file

    Returns:
        Batch job ID
    """
    # Upload the input file
    with open(input_file_path, "rb") as f:
        batch_input_file = client.files.create(file=f, purpose="batch")

    # Create the batch job
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint=BATCH_ENDPOINT,
        completion_window="24h",
        metadata={"description": "Supercasas apartment enrichment"},
    )

    return batch.id


def wait_for_batch(
    client: OpenAI,
    batch_id: str,
    poll_interval: int = 60,
    max_wait_time: int = 86400,  # 24 hours
) -> dict[str, Any]:
    """
    Wait for a batch job to complete.

    Args:
        client: OpenAI client instance
        batch_id: The batch job ID
        poll_interval: Seconds between status checks
        max_wait_time: Maximum seconds to wait

    Returns:
        Final batch status dictionary
    """
    start_time = time.time()

    while True:
        batch = client.batches.retrieve(batch_id)

        if batch.status == "completed":
            return {
                "status": "completed",
                "output_file_id": batch.output_file_id,
                "error_file_id": batch.error_file_id,
                "request_counts": batch.request_counts,
            }

        if batch.status in ("failed", "expired", "cancelled"):
            return {
                "status": batch.status,
                "error_file_id": batch.error_file_id,
            }

        elapsed = time.time() - start_time
        if elapsed > max_wait_time:
            return {"status": "timeout", "batch_id": batch_id}

        time.sleep(poll_interval)


def download_batch_results(
    client: OpenAI,
    output_file_id: str,
    output_path: str = BATCH_OUTPUT_FILE,
) -> str:
    """
    Download batch results from OpenAI.

    Args:
        client: OpenAI client instance
        output_file_id: The output file ID from completed batch
        output_path: Path to save the results

    Returns:
        Path to the downloaded file
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    content = client.files.content(output_file_id)

    with open(output_path, "wb") as f:
        f.write(content.read())

    return output_path


def parse_batch_results(
    output_path: str = BATCH_OUTPUT_FILE,
) -> dict[str, dict[str, Any]]:
    """
    Parse batch results JSONL file.

    Args:
        output_path: Path to the batch output JSONL file

    Returns:
        Dictionary mapping custom_id to enrichment data
    """
    results: dict[str, dict[str, Any]] = {}

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            result = json.loads(line)
            custom_id = result.get("custom_id", "")

            if result.get("response", {}).get("status_code") == 200:
                body = result["response"]["body"]
                choices = body.get("choices", [])

                if choices:
                    content = choices[0].get("message", {}).get("content", "{}")
                    try:
                        enrichment = json.loads(content)
                        results[custom_id] = enrichment
                    except json.JSONDecodeError:
                        results[custom_id] = {"_error": "Invalid JSON response"}
            else:
                results[custom_id] = {"_error": result.get("error", "Unknown error")}

    return results


def merge_enrichments(
    transformed_data: list[dict[str, Any]],
    enrichments: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Merge enrichment data back into transformed data.

    Args:
        transformed_data: Original transformed apartment data
        enrichments: Dictionary mapping custom_id to enrichment data

    Returns:
        List of enriched apartment dictionaries
    """
    enriched_data: list[dict[str, Any]] = []

    for row in transformed_data:
        site_id = row.get("site_id", "unknown")
        custom_id = f"apt-{site_id}"

        enriched_row = row.copy()

        if custom_id in enrichments:
            enrichment = enrichments[custom_id]

            # Only add defined enrichment fields
            for field in ENRICHMENT_FIELDS:
                if field in enrichment:
                    enriched_row[field] = enrichment[field]

            # Update facilities if enriched
            if "facilities" in enrichment:
                enriched_row["facilities"] = enrichment["facilities"]
            if "amount_of_facilities" in enrichment:
                enriched_row["amount_of_facilities"] = enrichment[
                    "amount_of_facilities"
                ]

        enriched_data.append(enriched_row)

    return enriched_data


def enrich_supercasas_data(
    transformed_data: list[dict[str, Any]],
    wait_for_completion: bool = True,
) -> list[dict[str, Any]] | str:
    """
    Enrich transformed apartment data using OpenAI Batch API.

    Args:
        transformed_data: List of transformed apartment dictionaries
        wait_for_completion: If True, wait for batch to complete and return enriched data.
                            If False, return batch_id immediately.

    Returns:
        Enriched data list if wait_for_completion=True, otherwise batch_id
    """
    client = OpenAI()

    # Prepare batch input
    logger.info("Preparing batch input file...")
    input_path = prepare_batch_input(transformed_data=transformed_data)
    logger.info(f"Created batch input: {input_path}")

    # Submit batch
    logger.info("Submitting batch to OpenAI...")
    batch_id = submit_batch(client=client, input_file_path=input_path)
    logger.info(f"Batch submitted: {batch_id}")

    if not wait_for_completion:
        return batch_id

    # Wait for completion
    logger.info("Waiting for batch completion...")
    result = wait_for_batch(client=client, batch_id=batch_id)
    logger.info(f"Batch status: {result['status']}")
    logger.info(f"Batch result: {result}")

    if result["status"] != "completed":
        raise RuntimeError(f"Batch failed with status: {result['status']}")

    if not result.get("output_file_id"):
        # Check if there's an error file
        if result.get("error_file_id"):
            error_content = client.files.content(result["error_file_id"])
            logger.error(f"Batch errors: {error_content.read().decode()}")
        raise RuntimeError("Batch completed but no output file was generated")

    # Download results
    logger.info("Downloading batch results...")
    output_path = download_batch_results(
        client=client,
        output_file_id=result["output_file_id"],
    )
    logger.info(f"Results saved to: {output_path}")

    # Parse and merge
    logger.info("Parsing and merging enrichments...")
    enrichments = parse_batch_results(output_path=output_path)
    enriched_data = merge_enrichments(
        transformed_data=transformed_data,
        enrichments=enrichments,
    )
    logger.info(f"Enriched {len(enriched_data)} apartments")

    # Apply local geolocation mapping
    logger.info("Applying local geolocation mapping...")
    for row in enriched_data:
        location = row.get("location", "")
        lat, lon, city_validated = get_santiago_coordinates(location)
        row["latitude"] = lat
        row["longitude"] = lon
        row["city_validated"] = city_validated

    # Filter out apartments not validated as Santiago, RD
    validated_data = filter_validated_locations(enriched_data=enriched_data)
    logger.info(f"Validated {len(validated_data)} apartments in Santiago, RD")

    return validated_data


def enrich_geolocation_local(row: dict[str, Any]) -> dict[str, Any]:
    """
    Enrich a row with latitude, longitude, and city_validated using local mapping.
    """
    location = row.get("location", "")
    lat, lon, city_validated = get_santiago_coordinates(location)
    row["latitude"] = lat
    row["longitude"] = lon
    row["city_validated"] = city_validated
    return row


def enrich_supercasas_data_local(
    transformed_data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Enrich transformed apartment data using local geolocation mapping.
    Args:
        transformed_data: List of transformed apartment dictionaries
    Returns:
        Enriched data list with geolocation fields
    """
    enriched = [enrich_geolocation_local(row.copy()) for row in transformed_data]
    # Puedes filtrar aquí si solo quieres los validados:
    # enriched = [row for row in enriched if row["city_validated"]]
    return enriched


def filter_validated_locations(
    enriched_data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Filter enriched data to only include apartments validated as being in Santiago, RD.

    Args:
        enriched_data: List of enriched apartment dictionaries

    Returns:
        Filtered list with only validated locations
    """
    validated: list[dict[str, Any]] = []
    discarded: list[dict[str, Any]] = []

    for row in enriched_data:
        city_validated = row.get(
            "city_validated", True
        )  # Default to True for backwards compatibility

        if city_validated:
            validated.append(row)
        else:
            discarded.append(row)
            logger.warning(
                f"Discarding apartment {row.get('site_id', 'unknown')} - "
                f"location '{row.get('location', 'unknown')}' not validated as Santiago, RD"
            )

    if discarded:
        logger.info(
            f"Discarded {len(discarded)} apartments with invalid/unconfirmed locations"
        )

    return validated


if __name__ == "__main__":
    import pandas as pd

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger: logging.Logger = logging.getLogger(name=__name__)

    # Test limit - only scrape and process this many apartments
    TEST_LIMIT = 5

    # Load transformed data
    from extract.supercasas import extract_from_supercasas
    from transform.supercasas import transform_supercasas_data

    raw_data = extract_from_supercasas(max_apartments=TEST_LIMIT)
    transformed_data = transform_supercasas_data(raw_data=raw_data, save_to_csv=False)
    logger.info(msg=f"Testing with {len(transformed_data)} apartments")

    # Enrich with OpenAI
    enriched_data = enrich_supercasas_data(
        transformed_data=transformed_data,
        wait_for_completion=True,
    )

    # Save enriched data
    if isinstance(enriched_data, list):
        df = pd.DataFrame(data=enriched_data)
        df.to_csv(ENRICHED_CSV_FILE, index=False)
        logger.info(msg=f"Saved enriched data to {ENRICHED_CSV_FILE}")
