"""
Extract module for Supercasas.com

This module is responsible ONLY for:
- Scraping raw HTML data from the website
- Basic field mapping to extract text content
- Saving raw data without any transformations

All data cleaning, type conversions, currency conversions, and business logic
should be handled in the transform module.
"""

import re
import logging
from typing import LiteralString
from bs4 import BeautifulSoup, Tag
from common.constants import HEADERS
from common.types import RequestParams
from common.utils import fetch_html

logger: logging.Logger = logging.getLogger(name=__name__)

BASE_URL = "https://www.supercasas.com"
PAGE_SKIP_QUERY_NAME = "PagingPageSkip"
PAGE_SKIPS = 0
LIMIT_OF_APARTMENTS_PER_PAGE = 100  # 100 for no limit
WEB_URL: LiteralString = f"{BASE_URL}/buscar/?do=2&ObjectType=123&PriceType=401&Locations=10095&PriceFrom=0.00&PriceTo=200000.00&SizeLotFrom=0&SizeLotTo=25000"

REQUEST_PARAMS: RequestParams = {
    "url": WEB_URL,
    "headers": HEADERS,
    "timeout": None,
}

IMPORTANT_CLASSES: dict[str, set[str]] = {
    "main_info": {"detail-ad-info-specs-block", "main-info"},
    "secondary_info": {"detail-ad-info-specs-block", "secondary-info"},
    "general_info": {"detail-ad-info-specs-block"},
    "details_info": {"detail-ad-info-specs-block"},
    "observations_info": {"detail-ad-info-specs-block"},
}

KEY_WORDS_MAPPING: dict[str, str] = {
    "rent": "Alquiler:",
    "rent_with_utilities": "Alquiler Amueblado:",
    "rooms": "habitaciones",
    "baths": "baños",
    "parking": "parqueos",
    "location": "Localización:",
    "previous_owners": "Condición:",
    "construction_meters": "Construcción:",
    "floor": "Nivel/Piso:",
    "is_modificable": "Edificable:",
    "use_type": "Uso Actual:",
    "land_meters": "Terreno:",
    "amount_of_elevators": "Ascensores:",
    "year_built": "Año Construcción:",
    "facilities": "Comodidades:",
    "observations": "Observaciones:",
}


def get_main_info(div: Tag) -> dict[str, str | None]:
    """Extract main info as raw strings without any transformation."""
    data: dict[str, str | None] = {}

    for child in div.find_all(name="div", recursive=False):
        span: Tag | None = child.find(name="span")
        if not span:
            continue

        raw: str = child.get_text(separator=" ", strip=True)

        for key, label in KEY_WORDS_MAPPING.items():
            if label not in raw:
                continue

            if key in ("rent", "rent_with_utilities"):
                data[key] = raw
            else:
                data[key] = raw.replace(label, "").strip()

    return data


def get_secondary_info(div: Tag) -> dict[str, str | None]:
    """Extract secondary info as raw strings without numeric conversion."""
    data: dict[str, str | None] = {}

    for child in div.find_all(name="div", recursive=False):
        span: Tag | None = child.find(name="span")
        if not span:
            continue

        raw: str = span.get_text(separator=" ", strip=True)

        for key, label in KEY_WORDS_MAPPING.items():
            if label.rstrip(":").lower() not in raw.lower():
                continue

            data[key] = raw
            break

    return data


def get_general_info(div: Tag) -> dict[str, str | None]:
    """Extract general info as raw strings without any parsing."""
    data: dict[str, str | None] = {}

    table: Tag | None = div.find(name="table")
    if not table:
        return data

    for row in table.find_all(name="tr"):
        cells = row.find_all(name="td")
        if len(cells) < 2:
            continue

        i = 0
        while i < len(cells) - 1:
            label_tag: Tag | None = cells[i].find(name="label")
            if not label_tag:
                i += 1
                continue

            label_text: str = label_tag.get_text(strip=True)
            value_text: str = cells[i + 1].get_text(separator=" ", strip=True)

            for key, label in KEY_WORDS_MAPPING.items():
                if label != label_text:
                    continue

                # Store raw string value, no parsing
                data[key] = value_text
                break

            i += 2

    return data


def get_details_info(div: Tag) -> dict[str, list[str]]:
    """Extract facilities list without derived calculations."""
    data: dict[str, list[str]] = {}

    ul: Tag | None = div.find(name="ul")
    if not ul:
        return data

    facilities: list[str] = []
    for li in ul.find_all(name="li"):
        text = li.get_text(strip=True)
        if text:
            facilities.append(text)

    data["facilities"] = facilities

    return data


def get_observations_info(div: Tag) -> dict[str, str | None]:
    """Extract observations as raw string."""
    data: dict[str, str | None] = {}

    h3: Tag | None = div.find(name="h3")
    if not h3 or h3.get_text(strip=True) != "Observaciones:":
        return data

    p: Tag | None = div.find(name="p")
    if not p:
        return data

    data["observations"] = p.get_text(separator="\n", strip=True)
    return data


def has_next_page(html: BeautifulSoup) -> bool:
    pager: Tag | None = html.select_one(
        selector="#bigsearch-results-inner-lowerbar-pages ul"
    )
    if not pager:
        return False

    active: Tag | None = pager.select_one(selector="li.active")
    if not active:
        return False

    next_li: Tag | None = active.find_next_sibling(name="li")
    if not next_li:
        return False

    a: Tag | None = next_li.find(name="a")
    if not a:
        return False

    return a.get_text(strip=True).isdigit()


def extract_apartment_data(apt_details: Tag) -> dict[str, object] | None:
    """
    Extract raw apartment data from HTML without any transformations.

    Returns a dictionary with raw string values. All type conversions,
    currency conversions, and derived fields are handled in the transform stage.
    """
    data: dict[str, object] = {
        "url": None,
        "site_id": None,
        # Raw price strings (e.g., "Alquiler: RD$ 25,000"
        "rent": None,
        "rent_with_utilities": None,
        # Raw numeric strings
        "rooms": None,
        "baths": None,
        "parking": None,
        # Location info (raw strings)
        "location": None,
        # General info (raw strings)
        "previous_owners": None,
        "construction_meters": None,
        "floor": None,
        "is_modificable": None,
        "use_type": None,
        "land_meters": None,
        "amount_of_elevators": None,
        "year_built": None,
        # List of facility strings
        "facilities": [],
        # Raw observations text
        "observations": None,
    }

    for div in apt_details.find_all(name="div", recursive=False):
        div_classes: set[str] = set(div.get(key="class") or [])
        if not div_classes:
            continue

        match div_classes:
            case x if x == IMPORTANT_CLASSES["main_info"]:
                main_info = get_main_info(div=div)
                data.update(main_info)
            case x if x == IMPORTANT_CLASSES["secondary_info"]:
                secondary_info = get_secondary_info(div=div)
                data.update(secondary_info)
            case x if x == IMPORTANT_CLASSES["general_info"]:
                general_info = get_general_info(div=div)
                details_info = get_details_info(div=div)
                observations_info = get_observations_info(div=div)

                data.update(general_info)
                data.update(details_info)
                data.update(observations_info)
            case _:
                continue

    return data


def scrape(max_apartments: int | None = None) -> list[dict[str, object]]:
    """Scrape raw apartment data from supercasas.com.

    Args:
        max_apartments: Maximum number of apartments to scrape. None for no limit.
    """
    page = 0
    apts_data: list[dict[str, object]] = []

    while True:
        # Check if we've reached the limit
        if max_apartments and len(apts_data) >= max_apartments:
            break

        page_url: str = f"{WEB_URL}&{PAGE_SKIP_QUERY_NAME}={page}"
        html, html_err = fetch_html(fetch_params={**REQUEST_PARAMS, "url": page_url})
        if html_err or html is None:
            break

        all_apartments_links: list[str] = [
            f"{BASE_URL}{a['href']}"
            for a in html.select(
                selector="#bigsearch-results-inner-results ul li a[href]"
            )
        ]

        for link in all_apartments_links[:LIMIT_OF_APARTMENTS_PER_PAGE]:
            # Check limit before each apartment
            if max_apartments and len(apts_data) >= max_apartments:
                break

            soup, err = fetch_html(fetch_params={**REQUEST_PARAMS, "url": link})
            if err or soup is None:
                continue

            apt_details: Tag | None = soup.select_one(selector="#detail-ad-info-specs")
            if not apt_details:
                continue

            apt_data: dict[str, object] | None = extract_apartment_data(
                apt_details=apt_details
            )
            if apt_data is None:
                continue

            apt_data["url"] = link
            match_site_id: re.Match[str] | None = re.search(
                pattern=r"/(\d+)(?:/)?$", string=link
            )
            if match_site_id:
                apt_data["site_id"] = match_site_id.group(1)

            apts_data.append(apt_data)

        if not has_next_page(html=html):
            break

        page += 1

    logger.info(msg=f"TOTAL APARTMENTS SCRAPED: {len(apts_data)}")
    return apts_data


def extract_from_supercasas(
    max_apartments: int | None = None,
) -> list[dict[str, object]]:
    """
    Extract raw apartment data from supercasas.com.

    This function only performs scraping and basic field extraction.
    No transformations, type conversions, or derived calculations are done here.

    Args:
        max_apartments: Maximum number of apartments to scrape. None for no limit.

    Returns:
        List of raw apartment dictionaries with string values.
    """
    logger.info(msg="STARTING THE SCRAPING PROCESS")
    data: list[dict[str, object]] = scrape(max_apartments=max_apartments)
    logger.info(msg=f"SCRAPING PROCESS COMPLETED - {len(data)} apartments extracted")
    return data


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger: logging.Logger = logging.getLogger(name=__name__)
    data = extract_from_supercasas()

    import pandas as pd

    df = pd.DataFrame(data=data)
    df.to_csv("supercasas_raw_data.csv", index=False)
