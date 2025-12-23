import re
import logging
import pandas as pd
from pathlib import Path
from typing import LiteralString
from bs4 import BeautifulSoup, Tag
from common.constants import HEADERS, USD_TO_DOP_RATE
from common.types import RequestParams
from common.utils import fetch_html

BASE_URL = "https://www.supercasas.com"
US_THRESHOLD = 10_000.00
RD_THRESHOLD_FOR_RENT = 1_000_000.00
PAGE_SKIP_QUERY_NAME = "PagingPageSkip"
PAGE_SKIPS = 0
LIMIT_OF_APARTMENTS_PER_PAGE = 1  # 100 for no limit
WEB_URL: LiteralString = f"{BASE_URL}/buscar/?do=2&ObjectType=123&PriceType=401&Locations=10095&PriceFrom=0.00&PriceTo=200000.00&SizeLotFrom=0&SizeLotTo=25000"
CSV_FILE_NAME = "data/supercasas_data.csv"

REQUEST_PARAMS: RequestParams = {
    "url": WEB_URL,
    "headers": HEADERS,
    "timeout": None,
}

ZERO_AS_NULL_FIELDS: set[str] = {
    "construction_meters",
    "land_meters",
}
STRING_NULL_VALUES: set[str] = {"n/d", "no informado", ""}

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


def get_main_info(div: Tag) -> dict[str, object]:
    data: dict[str, object] = {}

    for child in div.find_all(name="div", recursive=False):
        span: Tag | None = child.find(name="span")
        if not span:
            continue

        raw: str = child.get_text(separator=" ", strip=True)

        for key, label in KEY_WORDS_MAPPING.items():
            if label not in raw:
                continue

            if key in ("rent", "rent_with_utilities"):
                currency = None
                match_amount = None

                if "RD$" in raw:
                    currency = "DOP"
                    match_amount: re.Match[str] | None = re.search(
                        pattern=r"RD\$\s*([\d,]+)", string=raw
                    )
                elif "US$" in raw or "USD" in raw:
                    currency = "USD"
                    match_amount = re.search(
                        pattern=r"(?:US\$|USD)\s*([\d,]+)", string=raw
                    )

                if not match_amount:
                    continue

                value = float(match_amount.group(1).replace(",", ""))
                data["original_rent"] = value
                if currency == "USD":
                    data["original_rent"] = value
                    value *= USD_TO_DOP_RATE

                data[key] = value
                data["origin_currency"] = currency

            else:
                data[key] = raw.replace(label, "").strip()

    return data


def get_secondary_info(div: Tag) -> dict[str, object]:
    data: dict[str, object] = {}

    for child in div.find_all(name="div", recursive=False):
        span: Tag | None = child.find(name="span")
        if not span:
            continue

        raw: str = span.get_text(separator=" ", strip=True).lower()

        match: re.Match[str] | None = re.search(pattern=r"(\d+(?:\.\d+)?)", string=raw)
        if not match:
            continue

        numeric = match.group(1)

        for key, label in KEY_WORDS_MAPPING.items():
            if label.rstrip(":").lower() not in raw:
                continue

            if key in ("rooms", "parking"):
                data[key] = int(float(numeric))
            else:
                data[key] = float(numeric)

            break

    return data


def get_general_info(div: Tag) -> dict[str, object]:
    data: dict[str, object] = {}

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

                if key == "location" and ">" in value_text:
                    city, location = map(
                        str.strip, value_text.split(sep=">", maxsplit=1)
                    )
                    data["city"] = city
                    data[key] = location

                elif key in ("land_meters", "construction_meters"):
                    match: re.Match[str] | None = re.search(
                        pattern=r"(\d+(?:\.\d+)?)", string=value_text
                    )
                    if match:
                        data[key] = float(match.group(1))

                elif key == "amount_of_elevators":
                    data[key] = int(value_text)
                else:
                    data[key] = value_text

                break

            i += 2

    return data


def get_details_info(div: Tag) -> dict[str, object]:
    data: dict[str, object] = {}

    ul: Tag | None = div.find(name="ul")
    if not ul:
        return data

    facilities = data.setdefault("facilities", [])
    assert isinstance(facilities, list)

    for li in ul.find_all(name="li"):
        text = li.get_text(strip=True)
        if not text:
            continue

        facilities.append(text)

    data["amount_of_facilities"] = len(facilities)

    return data


def get_observations_info(div: Tag) -> dict[str, object]:
    data: dict[str, object] = {}

    h3: Tag | None = div.find(name="h3")
    if not h3 or h3.get_text(strip=True) != "Observaciones:":
        return data

    p: Tag | None = div.find(name="p")
    if not p:
        return data

    data["observations"] = p.get_text(separator="\n", strip=True)
    return data


def normalize_zero_values(data: dict[str, object]) -> dict[str, object]:
    """
    Normalizes placeholder values to None:
    - 0 / 0.0 for selected numeric fields where zero means 'not informed'
    - 'N/D', 'No informado', empty strings for string fields
    """

    # Numeric fields where 0 means "not informed"
    for field in ZERO_AS_NULL_FIELDS:
        value = data.get(field)
        if isinstance(value, (int, float)) and value == 0:
            data[field] = None

    for key, value in data.items():
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned.lower() in STRING_NULL_VALUES:
                data[key] = None

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
    data: dict[str, object] = {
        "url": None,
        "site_id": None,
        "original_rent": None,
        "rent": None,
        "rent_with_utilities": None,
        "origin_currency": None,
        "rooms": None,
        "baths": None,
        "parking": None,
        "location": None,
        "city": None,
        "previous_owners": None,
        "construction_meters": None,
        "floor": None,
        "is_modificable": None,
        "use_type": None,
        "land_meters": None,
        "amount_of_elevators": None,
        "year_built": None,
        "facilities": list[str],
        "amount_of_facilities": None,
        "observations": None,
    }

    for div in apt_details.find_all(name="div", recursive=False):
        div_classes: set[str] = set(div.get(key="class") or [])
        if not div_classes:
            continue

        match div_classes:
            case x if x == IMPORTANT_CLASSES["main_info"]:
                main_info: dict[str, object] = get_main_info(div=div)
                data.update(main_info)
            case x if x == IMPORTANT_CLASSES["secondary_info"]:
                secondary_info: dict[str, object] = get_secondary_info(div=div)
                data.update(secondary_info)
            case x if x == IMPORTANT_CLASSES["general_info"]:
                general_info: dict[str, object] = get_general_info(div=div)
                details_info: dict[str, object] = get_details_info(div=div)
                observations_info: dict[str, object] = get_observations_info(div=div)

                data.update(general_info)
                data.update(details_info)
                data.update(observations_info)
            case _:
                continue

    data = normalize_zero_values(data=data)
    return data


def scrape() -> list[dict[str, object]] | None:
    page = 0
    apts_data: list[dict[str, object]] = []

    while True:
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


def main() -> list[dict[str, object]] | None:
    logger.info(msg="STARTING THE SCRAPING PROCESS")
    data: list[dict[str, object]] | None = scrape()
    logger.info(msg="SCRAPING PROCESS COMPLETED")

    df = pd.DataFrame(data=data)
    logger.info(msg=f"DATAFRAME SHAPE: {df.shape}")

    Path(CSV_FILE_NAME).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_or_buf=CSV_FILE_NAME, index=False)

    return data


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger: logging.Logger = logging.getLogger(name=__name__)
    main()
