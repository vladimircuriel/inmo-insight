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
