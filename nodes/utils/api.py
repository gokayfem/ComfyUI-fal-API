"""fal.ai API submission helpers for ComfyUI-fal-API."""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any, Callable, NoReturn

from .config import FalConfig
from .errors import FalApiError, extract_error_message, raise_fal_error
from .logger import logger

_MAX_QUEUE_LOG_LINES = 10_000


def _check_interruption() -> None:
    """Raise ComfyUI's InterruptProcessingException if the user cancelled.

    A no-op when running outside ComfyUI.
    """
    try:
        import comfy.model_management as model_management
    except ImportError:
        return
    model_management.throw_exception_if_processing_interrupted()


def _is_interruption(exc: BaseException) -> bool:
    """Detect ComfyUI's interruption exception without importing comfy."""
    return exc.__class__.__name__ == "InterruptProcessingException"


def _log_message_from_entry(entry: Any) -> str | None:
    """Extract a printable message from a fal queue log entry."""
    if isinstance(entry, dict):
        message = entry.get("message")
        return str(message) if message else None
    text = str(entry)
    return text if text else None


def _make_queue_callback(endpoint: str) -> Callable[[Any], None]:
    """Build an on_queue_update callback that logs progress and honors cancel."""
    import fal_client

    seen_lines: set = set()
    last_position: list[int | None] = [None]

    def on_queue_update(status: Any) -> None:
        # Anything raised here (interruption) must propagate to the caller.
        _check_interruption()

        if isinstance(status, fal_client.InProgress):
            for entry in status.logs or []:
                message = _log_message_from_entry(entry)
                if message and message not in seen_lines:
                    if len(seen_lines) < _MAX_QUEUE_LOG_LINES:
                        seen_lines.add(message)
                    logger.info("[%s] %s", endpoint, message)
        elif isinstance(status, fal_client.Queued):
            position = getattr(status, "position", None)
            if position != last_position[0]:
                last_position[0] = position
                logger.info("[%s] queued (position %s)", endpoint, position)

    return on_queue_update


async def _submit_multiple_async(
    endpoint: str, arguments: dict[str, Any], variations: int
) -> list[Any]:
    """Submit multiple jobs concurrently and gather results (with exceptions).

    Interruption is only observed between the submit and gather phases — the
    per-request polling here has no queue callback, so a ComfyUI Cancel takes
    effect once the in-flight variations settle (known limitation).
    """
    from fal_client import AsyncClient

    # Validate the key via get_client() first so a missing/placeholder key
    # raises the actionable config error instead of a raw auth failure.
    FalConfig().get_client()
    client = AsyncClient(key=FalConfig().get_key())

    def variation_arguments(index: int) -> dict[str, Any]:
        if "seed" in arguments:
            return {**arguments, "seed": arguments.get("seed", 0) + index}
        return arguments

    async def submit_and_get(index: int) -> Any:
        handler = await client.submit(endpoint, arguments=variation_arguments(index))
        return await handler.get()

    # One flow per variation so a single submit failure only loses that
    # variation instead of failing the whole batch.
    return await asyncio.gather(
        *[submit_and_get(i) for i in range(variations)], return_exceptions=True
    )


def _partition_results(
    endpoint: str, raw_results: list[Any]
) -> tuple[list[Any], list[tuple]]:
    """Split gathered results into successes and logged failures."""
    successes: list[Any] = []
    failures: list[tuple] = []
    for index, item in enumerate(raw_results):
        if isinstance(item, BaseException):
            message, status_code = extract_error_message(item)
            logger.error("[%s] variation %d failed: %s", endpoint, index, message)
            failures.append((index, message, status_code))
        else:
            successes.append(item)
    return successes, failures


def _raise_generation_error(model_name: str, error: Exception | str) -> NoReturn:
    """Normalize an exception or error string into a raised FalApiError."""
    if isinstance(error, BaseException):
        if _is_interruption(error) or not isinstance(error, Exception):
            raise error
        raise_fal_error(model_name, error)
    raise FalApiError(model_name, str(error))


class ApiHandler:
    """Utility functions for fal.ai API interactions."""

    @staticmethod
    def submit_and_get_result(
        endpoint: str,
        arguments: dict[str, Any],
        timeout: float | None = None,
    ) -> Any:
        """Submit a job via client.subscribe and return the final result.

        Logs queue position and in-progress log lines, and checks for ComfyUI
        interruption on every queue update. ``timeout`` is reserved for future
        use (fal_client 1.0 subscribe does not accept one).
        """
        del timeout  # Reserved; not supported by fal_client 1.0 subscribe.
        client = FalConfig().get_client()
        callback = _make_queue_callback(endpoint)
        try:
            return client.subscribe(
                endpoint,
                arguments=arguments,
                with_logs=True,
                on_queue_update=callback,
            )
        except FalApiError:
            raise
        except Exception as exc:
            if _is_interruption(exc):
                raise
            raise_fal_error(endpoint, exc)

    @staticmethod
    def submit_multiple_and_get_results(
        endpoint: str, arguments: dict[str, Any], variations: int
    ) -> list[Any]:
        """Submit multiple variations concurrently and return successful results.

        Failed variations are logged; raises FalApiError only if ALL fail.
        """
        try:
            # Run the async code in a dedicated thread to avoid event loop
            # conflicts with ComfyUI's own loop.
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    asyncio.run,
                    _submit_multiple_async(endpoint, arguments, variations),
                )
                raw_results = future.result()
        except FalApiError:
            raise
        except Exception as exc:
            if _is_interruption(exc):
                raise
            raise_fal_error(endpoint, exc)

        successes, failures = _partition_results(endpoint, raw_results)
        if not successes:
            first_message = failures[0][1] if failures else "no results returned"
            first_status = failures[0][2] if failures else None
            raise FalApiError(
                endpoint,
                f"All {variations} variations failed: {first_message}",
                first_status,
            )
        return successes

    @staticmethod
    def handle_video_generation_error(
        model_name: str, error: Exception | str
    ) -> NoReturn:
        """Raise a normalized FalApiError for a video generation failure."""
        _raise_generation_error(model_name, error)

    @staticmethod
    def handle_image_generation_error(
        model_name: str, error: Exception | str
    ) -> NoReturn:
        """Raise a normalized FalApiError for an image generation failure."""
        _raise_generation_error(model_name, error)

    @staticmethod
    def handle_text_generation_error(
        model_name: str, error: Exception | str
    ) -> NoReturn:
        """Raise a normalized FalApiError for a text generation failure."""
        _raise_generation_error(model_name, error)
