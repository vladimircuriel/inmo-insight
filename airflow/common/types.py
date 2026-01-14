from typing import TypedDict


class RequestParams(TypedDict):
    url: str
    headers: dict[str, str] | None
    timeout: int | None
