"""Shared async HTTP access with bounded retries and exponential backoff.

Every outbound call in Athena goes through :func:`fetch_json` or
:func:`fetch_text` so that timeouts, rate limits and transient 5xx responses are
handled identically and logged in one place.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

import httpx

from ..config import get_settings
from ..errors import ExternalAPIError
from ..logging_setup import get_logger, log_event

logger = get_logger("http")

# Status codes that are worth trying again.
RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


def _user_agent() -> str:
    settings = get_settings()
    return f"Athena-Research-Agent/1.0 (mailto:{settings.contact_email})"


async def _request(
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    settings = get_settings()
    attempts = settings.max_tool_retries
    merged_headers = {"User-Agent": _user_agent(), **(headers or {})}
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            async with httpx.AsyncClient(
                timeout=settings.request_timeout_seconds, follow_redirects=True
            ) as client:
                response = await client.request(
                    method, url, params=params, headers=merged_headers
                )
            if response.status_code in RETRYABLE_STATUS:
                raise httpx.HTTPStatusError(
                    f"retryable status {response.status_code}",
                    request=response.request,
                    response=response,
                )
            response.raise_for_status()
            return response

        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            last_error = exc
            non_retryable = (
                isinstance(exc, httpx.HTTPStatusError)
                and exc.response.status_code not in RETRYABLE_STATUS
            )
            if non_retryable or attempt == attempts:
                break
            # Exponential backoff with jitter, so parallel searches don't
            # synchronise and hammer a rate-limited endpoint together.
            delay = (2 ** (attempt - 1)) + random.uniform(0, 0.4)
            log_event(
                logger,
                f"retrying {url} in {delay:.1f}s",
                attempt=attempt,
                max_attempts=attempts,
                error=str(exc),
            )
            await asyncio.sleep(delay)

    raise ExternalAPIError(url, f"request failed after {attempts} attempts: {last_error}")


async def fetch_json(
    url: str, *, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    response = await _request("GET", url, params=params)
    try:
        return response.json()
    except ValueError as exc:
        raise ExternalAPIError(url, f"response was not valid JSON: {exc}") from exc


async def fetch_text(url: str, *, params: dict[str, Any] | None = None) -> str:
    response = await _request("GET", url, params=params)
    return response.text


async def fetch_bytes(url: str) -> bytes:
    response = await _request("GET", url)
    return response.content
