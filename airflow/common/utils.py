import logging
import time
from bs4 import BeautifulSoup
import requests

from common.types import RequestParams

logger: logging.Logger = logging.getLogger(name=__name__)


def fetcher(
    fetch_params: RequestParams,
) -> tuple[requests.Response | None, Exception | None]:
    url: str = fetch_params["url"]
    headers: dict | None = fetch_params.get("headers")
    delay: int = fetch_params.get("delay", 15)

    try:
        response: requests.Response = requests.get(url=url, headers=headers, timeout=15)

        if response.status_code == 429:
            time.sleep(delay)
            logger.info(msg=f"RETRYING REQUEST AFTER DELAY ({url})")
            response = requests.get(url=url, headers=headers, timeout=15)

        response.raise_for_status()
        logger.info(msg=f"SUCCESSFUL REQUEST ({url})")

        return response, None

    except Exception as e:
        logger.info(msg=f"ERROR DURING REQUEST ({url}): {e}")
        return None, e


def parse_html(html_string: str) -> BeautifulSoup:
    """Converts an HTML string into a BeautifulSoup object.
    Args:
        html_string (str): The HTML content as a string.
    Returns:
        BeautifulSoup: A BeautifulSoup object containing the parsed HTML content.
    """
    return BeautifulSoup(markup=html_string, features="html.parser")


def fetch_html(
    fetch_params: RequestParams,
) -> tuple[BeautifulSoup | None, Exception | None]:
    """Fetches the HTML content from the specified URL and returns a BeautifulSoup object.
    Args:
        url (str): The URL to fetch the HTML content from.
        headers (dict): The headers to include in the GET request.
    Returns:
        BeautifulSoup | None: A BeautifulSoup object containing the parsed HTML content, or None if an error occurred.
    """
    url: str = fetch_params["url"]

    response, error = fetcher(fetch_params=fetch_params)
    if error or response is None:
        logger.info(msg=f"ERROR DURING REQUEST FOR SOUP ({url}): {error}")
        return None, error

    html: BeautifulSoup = parse_html(html_string=response.text)
    return html, None


# Currency exchange API
FX_RATES_API_URL = "https://api.fxratesapi.com/latest"
FALLBACK_USD_TO_DOP = 64.0  # Fallback rate if API fails


def get_usd_to_dop_rate() -> float:
    """
    Fetch current USD to DOP exchange rate from fxratesapi.com.

    Returns:
        float: The USD to DOP exchange rate, or fallback value if API fails.
    """
    try:
        response = requests.get(url=FX_RATES_API_URL, timeout=10)
        response.raise_for_status()

        data = response.json()

        if not data.get("success", False):
            logger.warning("FX API responded with success=false, using fallback rate")
            return FALLBACK_USD_TO_DOP

        rates = data.get("rates", {})
        dop_rate = rates.get("DOP")

        if dop_rate is None:
            logger.warning("DOP rate not found in API response, using fallback rate")
            return FALLBACK_USD_TO_DOP

        # API returns rates relative to EUR base, so we need USD to DOP
        usd_rate = rates.get("USD")
        if usd_rate is None or usd_rate == 0:
            logger.warning("USD rate not found in API response, using fallback rate")
            return FALLBACK_USD_TO_DOP

        # Calculate USD to DOP: DOP/USD
        usd_to_dop = dop_rate / usd_rate

        logger.info(f"Fetched USD to DOP rate: {usd_to_dop:.2f}")
        return usd_to_dop

    except Exception as e:
        logger.warning(f"Failed to fetch exchange rate: {e}, using fallback rate")
        return FALLBACK_USD_TO_DOP
