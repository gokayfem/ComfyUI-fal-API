"""fal.ai API submission helpers for ComfyUI-fal-API."""

from __future__ import annotations

import asyncio
import concurrent.futures
import time
from typing import Any, Callable, NoReturn

from .config import FalConfig
from .errors import FalApiError, extract_error_message, raise_fal_error
from .job_store import JobStore
from .ledger import SessionLedger
from .logger import logger
from .pricing import PricingUtils
from .result_cache import ResultCache

_RECOVERY_POLL_INTERVAL_S = 0.5

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


def _record_ledger_entry(
    endpoint: str,
    request_id: str | None,
    duration_s: float,
    est_cost_override: float | None = None,
    free: bool = False,
) -> None:
    """Record one fal call in the session ledger.

    ``free=True`` marks a recovery/replay that spent no new money (est_cost
    None). Best-effort bookkeeping: any pricing or ledger error is swallowed
    so it can never break generation.
    """
    if free:
        est_cost = est_cost_override
    else:
        try:
            est_cost = PricingUtils.estimate(endpoint, 1)["total"]
        except Exception as exc:
            logger.debug("[%s] cost estimation failed: %s", endpoint, exc)
            est_cost = None
    try:
        SessionLedger().record(endpoint, request_id, duration_s, est_cost)
    except Exception as exc:
        logger.debug("[%s] ledger record failed: %s", endpoint, exc)


def _spend_guard_preflight(endpoint: str) -> None:
    """Enforce the spend budget before submitting a paid call.

    A no-op when the billing module is absent. A FalApiError raised by
    SpendGuard (over budget) propagates to the caller.
    """
    try:
        from .billing import SpendGuard
    except ImportError:
        return
    SpendGuard.preflight(endpoint)


def _store_result_in_cache(
    endpoint: str,
    arguments: dict[str, Any],
    result: Any,
    request_id: str | None,
) -> None:
    """Persist a successful live result in the persistent cache.

    Best-effort bookkeeping: only dict results are cached and any cache
    error is swallowed so it can never break generation.
    """
    if not isinstance(result, dict):
        return
    try:
        ResultCache().put(endpoint, arguments, result, request_id)
    except Exception as exc:
        logger.debug("[%s] result cache store failed: %s", endpoint, exc)


def _remember_result_urls(endpoint: str, request_id: str | None, result: Any) -> None:
    """Best-effort provenance bookkeeping: map result URLs to their request."""
    if not request_id:
        return
    try:
        ResultCache().remember_urls(endpoint, request_id, result)
    except Exception as exc:
        logger.debug("[%s] remember_urls failed: %s", endpoint, exc)


def _finalize_live_call(endpoint: str, request_id: str | None, started: float) -> None:
    """Log the finished call and record it in the session ledger."""
    duration_s = time.monotonic() - started
    logger.info(
        "[%s] call finished in %.1fs (request_id=%s)",
        endpoint,
        duration_s,
        request_id,
    )
    _record_ledger_entry(endpoint, request_id, duration_s)


async def _close_async_client(client: Any) -> None:
    """Best-effort close of a per-call AsyncClient's underlying httpx client.

    fal_client.AsyncClient lazily caches an httpx.AsyncClient per instance
    (bound to the current event loop); we create one AsyncClient per call, so
    close it here to avoid leaking connections. Resolving ``_client`` does no
    network I/O; any failure is swallowed — cleanup must never mask a result
    or an error from the call itself.
    """
    try:
        httpx_client = await client._client
        await httpx_client.aclose()
    except Exception as exc:
        logger.debug("async fal client close failed: %s", exc)

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
        skip_cache: bool = False,
    ) -> Any:
        """Submit a job via client.subscribe and return the final result.

        Checks the spend budget first, then the persistent result cache: an
        identical previous call returns its stored result immediately (no
        charge, no ledger entry). Pass ``skip_cache=True`` to force a live
        call (e.g. force_rerun). Logs queue position and in-progress log
        lines, and checks for ComfyUI interruption on every queue update.
        ``timeout`` is reserved for future use (fal_client 1.0 subscribe does
        not accept one).
        """
        del timeout  # Reserved; not supported by fal_client 1.0 subscribe.

        # Cache first: a hit costs nothing, so it must not be blocked by the
        # spend guard (which only gates live, billable calls).
        if not skip_cache:
            cached = ResultCache().get(endpoint, arguments)
            if cached is not None:
                return cached

        _spend_guard_preflight(endpoint)

        client = FalConfig().get_client()
        callback = _make_queue_callback(endpoint)
        request_id_ref: list[str | None] = [None]

        def on_enqueue(request_id: str) -> None:
            request_id_ref[0] = request_id

        started = time.monotonic()
        try:
            result = client.subscribe(
                endpoint,
                arguments=arguments,
                with_logs=True,
                on_enqueue=on_enqueue,
                on_queue_update=callback,
            )
        except FalApiError:
            raise
        except Exception as exc:
            if _is_interruption(exc):
                raise
            raise_fal_error(endpoint, exc)
        finally:
            _finalize_live_call(endpoint, request_id_ref[0], started)

        _store_result_in_cache(endpoint, arguments, result, request_id_ref[0])
        _remember_result_urls(endpoint, request_id_ref[0], result)
        return result

    @staticmethod
    async def submit_and_get_result_async(
        endpoint: str,
        arguments: dict[str, Any],
        skip_cache: bool = False,
    ) -> Any:
        """Async twin of ``submit_and_get_result`` for async-capable ComfyUI.

        Same semantics — spend-guard preflight, persistent result cache,
        queue-progress logging, interruption via the queue callback, ledger
        recording and cache/provenance bookkeeping — but awaits the fal call
        on the event loop so the executor can run other graph branches
        concurrently. The AsyncClient is created per call because its cached
        httpx client is bound to the current event loop (ComfyUI runs each
        prompt in a fresh loop via ``asyncio.run``).
        """
        # Cache first: a hit costs nothing, so it must not be blocked by the
        # spend guard (which only gates live, billable calls).
        if not skip_cache:
            cached = ResultCache().get(endpoint, arguments)
            if cached is not None:
                return cached

        # off-loop: preflight may make a blocking balance HTTP call
        await asyncio.to_thread(_spend_guard_preflight, endpoint)

        from fal_client import AsyncClient

        # Validate the key via get_client() first so a missing/placeholder key
        # raises the actionable config error instead of a raw auth failure.
        FalConfig().get_client()
        client = AsyncClient(key=FalConfig().get_key())
        callback = _make_queue_callback(endpoint)
        request_id_ref: list[str | None] = [None]

        def on_enqueue(request_id: str) -> None:
            request_id_ref[0] = request_id

        # The queue callback checks interruption on every update while the job
        # runs; this covers a cancel that landed before submission (and stays
        # outside the try so it cannot record a ledger entry for a job that
        # was never submitted).
        _check_interruption()

        started = time.monotonic()
        try:
            result = await client.subscribe(
                endpoint,
                arguments=arguments,
                with_logs=True,
                on_enqueue=on_enqueue,
                on_queue_update=callback,
            )
        except FalApiError:
            raise
        except Exception as exc:
            if _is_interruption(exc):
                raise
            raise_fal_error(endpoint, exc)
        finally:
            _finalize_live_call(endpoint, request_id_ref[0], started)
            await _close_async_client(client)

        _store_result_in_cache(endpoint, arguments, result, request_id_ref[0])
        _remember_result_urls(endpoint, request_id_ref[0], result)
        return result

    @staticmethod
    def submit_only(endpoint: str, arguments: dict[str, Any]) -> str:
        """Submit a job without waiting and return its request id.

        Checks the spend budget first (async fan-out must respect it too).
        Does not record to the session ledger — the collect side
        (``result_from_request_id``) records the call.
        """
        _spend_guard_preflight(endpoint)
        client = FalConfig().get_client()
        try:
            handle = client.submit(endpoint, arguments=arguments)
        except FalApiError:
            raise
        except Exception as exc:
            if _is_interruption(exc):
                raise
            raise_fal_error(endpoint, exc)
        logger.info("[%s] submitted async (request_id=%s)", endpoint, handle.request_id)
        # Best-effort bookkeeping: the persistent job inbox lets this request
        # be found and collected even after a ComfyUI restart.
        try:
            JobStore().record_submit(endpoint, handle.request_id)
        except Exception as exc:
            logger.debug("[%s] job store record_submit failed: %s", endpoint, exc)
        return handle.request_id

    @staticmethod
    def result_from_request_id(
        endpoint: str, request_id: str, record_cost: bool = True
    ) -> dict[str, Any]:
        """Wait for and fetch the result of a previously submitted request.

        Reconstructs a queue handle from the request id, polls until the
        request completes (honoring ComfyUI interruption), and returns the
        result payload. A request that already completed returns immediately
        without incurring new charges — the result-recovery path.
        """
        label = f"{endpoint}#{request_id}"
        client = FalConfig().get_client()
        started = time.monotonic()
        try:
            handle = client.get_handle(endpoint, request_id)
            for _status in handle.iter_events(
                with_logs=False, interval=_RECOVERY_POLL_INTERVAL_S
            ):
                _check_interruption()
            result = handle.get()
        except FalApiError:
            raise
        except Exception as exc:
            if _is_interruption(exc):
                raise
            raise_fal_error(label, exc)
        finally:
            duration_s = time.monotonic() - started
            logger.info(
                "[%s] result recovery finished in %.1fs (request_id=%s)",
                endpoint,
                duration_s,
                request_id,
            )
            # record_cost=False marks a pure recovery of an old request:
            # log the fetch for traceability but count no new spend.
            if record_cost:
                _record_ledger_entry(endpoint, request_id, duration_s)
            else:
                _record_ledger_entry(endpoint, request_id, duration_s, est_cost_override=None, free=True)
        # Best-effort bookkeeping: mark the job collected in the persistent
        # inbox (inserting it if it was submitted in another session).
        try:
            JobStore().mark_collected(request_id)
        except Exception as exc:
            logger.debug("[%s] job store mark_collected failed: %s", endpoint, exc)
        _remember_result_urls(endpoint, request_id, result)
        return result

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
