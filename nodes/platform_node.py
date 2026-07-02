"""Platform nodes: async submit/collect, result recovery, cost tools, media saving."""

from __future__ import annotations

import os
import shutil
from typing import Any
from urllib.parse import urlparse

from .dynamic.any_endpoint import build_overlay_arguments, extract_flexible_outputs
from .fal_utils import (
    ApiHandler,
    FalApiError,
    MediaUtils,
    PricingUtils,
    SessionLedger,
    logger,
)

_CATEGORY = "FAL/Platform"

# Custom ComfyUI type carried between FalSubmit and FalCollect.
# Shape: {"endpoint_id": str, "request_id": str}
FAL_HANDLE_TYPE = "FAL_HANDLE"

_FLEXIBLE_RETURN_TYPES = ("IMAGE", "VIDEO", "AUDIO", "STRING")
_FLEXIBLE_RETURN_NAMES = ("images", "video", "audio", "result_json")

_MAX_SEED = 2**31 - 1
_SAVE_COUNTER_LIMIT = 100_000


def _collect_result(
    node_name: str, endpoint_id: str, request_id: str, record_cost: bool = True
) -> tuple[Any, Any, Any, str]:
    """Fetch a queued result by id and extract flexible outputs from it.

    ``record_cost=False`` marks a pure recovery of a past request — the fetch
    is logged for traceability but adds no new spend to the session ledger.
    """
    endpoint = (endpoint_id or "").strip()
    request = (request_id or "").strip()
    if not endpoint or not request:
        raise FalApiError(node_name, "Both endpoint_id and request_id are required")
    result = ApiHandler.result_from_request_id(endpoint, request, record_cost=record_cost)
    return extract_flexible_outputs(result)


class FalSubmit:
    """Queue a fal.ai job without waiting; pair with Fal Collect for the result."""

    RETURN_TYPES = (FAL_HANDLE_TYPE, "STRING")
    RETURN_NAMES = ("handle", "request_id")
    FUNCTION = "submit"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Submit a job to any fal.ai endpoint and return immediately with a "
        "handle. Wire several Submits into Collects to run generations in "
        "parallel instead of one at a time."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "endpoint_id": (
                    "STRING",
                    {
                        "default": "fal-ai/kling-video/v3/pro/image-to-video",
                        "tooltip": (
                            "fal endpoint id, e.g. fal-ai/kling-video/v3/pro/image-to-video. "
                            "Submitting queues the job instantly, so multiple Fal Submit nodes "
                            "fan out in parallel; wire each handle into a Fal Collect node to "
                            "wait for its result."
                        ),
                    },
                ),
                "arguments_json": (
                    "STRING",
                    {
                        "default": "{}",
                        "multiline": True,
                        "tooltip": (
                            "JSON object of API arguments. Connected media inputs "
                            "and seed override matching keys here."
                        ),
                    },
                ),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Uploaded and sent as image_url"}),
                "image_2": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Second image; when set together with 'image', both are "
                            "also sent as image_urls [url1, url2]"
                        )
                    },
                ),
                "video": ("VIDEO", {"tooltip": "Uploaded and sent as video_url"}),
                "audio": ("AUDIO", {"tooltip": "Uploaded and sent as audio_url"}),
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": _MAX_SEED,
                        "control_after_generate": True,
                        "tooltip": "-1 = omit seed; any other value is sent to the API",
                    },
                ),
            },
        }

    def submit(
        self,
        endpoint_id: str,
        arguments_json: str = "{}",
        image: Any = None,
        image_2: Any = None,
        video: Any = None,
        audio: Any = None,
        seed: int = -1,
    ) -> tuple[dict[str, str], str]:
        endpoint = (endpoint_id or "").strip()
        if not endpoint:
            raise FalApiError("FalSubmit", "endpoint_id is required")

        arguments = build_overlay_arguments(
            endpoint, arguments_json, image, image_2, video, audio, seed
        )
        request_id = ApiHandler.submit_only(endpoint, arguments)
        logger.info("[%s] submitted request %s", endpoint, request_id)
        handle = {"endpoint_id": endpoint, "request_id": request_id}
        return (handle, request_id)


class FalCollect:
    """Wait for a queued fal.ai job (from Fal Submit) and extract its outputs."""

    RETURN_TYPES = _FLEXIBLE_RETURN_TYPES
    RETURN_NAMES = _FLEXIBLE_RETURN_NAMES
    FUNCTION = "collect"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Block until a job submitted with Fal Submit finishes, then extract "
        "outputs opportunistically. The raw result is always available as JSON."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "handle": (
                    FAL_HANDLE_TYPE,
                    {"tooltip": "Handle from a Fal Submit node"},
                ),
            },
        }

    def collect(self, handle: Any) -> tuple[Any, Any, Any, str]:
        if not isinstance(handle, dict) or not handle.get("endpoint_id") or not handle.get("request_id"):
            raise FalApiError(
                "FalCollect",
                "Expected a FAL_HANDLE dict with 'endpoint_id' and 'request_id' "
                "keys; connect the handle output of a Fal Submit node.",
            )
        return _collect_result("FalCollect", handle["endpoint_id"], handle["request_id"])


class FalResultByRequestId:
    """Fetch any past fal.ai generation by its request id, without re-paying."""

    RETURN_TYPES = _FLEXIBLE_RETURN_TYPES
    RETURN_NAMES = _FLEXIBLE_RETURN_NAMES
    FUNCTION = "fetch"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Recover a past generation by request id. Results are fetched from "
        "fal's queue by id, so nothing is re-generated or re-billed."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "endpoint_id": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "fal endpoint id the request was originally submitted to",
                    },
                ),
                "request_id": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Recover any past generation from your logs, the session cost "
                            "report, or the fal.ai dashboard without re-paying; the result "
                            "is fetched from fal's queue by this id."
                        ),
                    },
                ),
            },
        }

    def fetch(self, endpoint_id: str, request_id: str) -> tuple[Any, Any, Any, str]:
        return _collect_result(
            "FalResultByRequestId", endpoint_id, request_id, record_cost=False
        )


class FalCostEstimator:
    """Estimate the cost of running an endpoint N times, from registry pricing."""

    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("report", "total_usd")
    FUNCTION = "estimate"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Estimate USD cost for running a fal endpoint a number of times. "
        "Never fails: unknown endpoints produce a 'pricing unknown' report."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "endpoint_id": (
                    "STRING",
                    {
                        "default": "fal-ai/flux/dev",
                        "tooltip": (
                            "fal endpoint id to estimate. See the model list in the "
                            "README for available endpoint ids."
                        ),
                    },
                ),
                "runs": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 10000,
                        "tooltip": "Number of runs to estimate the total cost for",
                    },
                ),
            },
        }

    def estimate(self, endpoint_id: str, runs: int = 1) -> tuple[str, float]:
        endpoint = (endpoint_id or "").strip()
        try:
            est = PricingUtils.estimate(endpoint, runs=int(runs))
            report = PricingUtils.format_report(est)
            total = est.get("total")
        except Exception as err:  # This node must never fail the graph.
            logger.warning("FalCostEstimator: could not estimate %s: %s", endpoint, err)
            report = f"{endpoint or '(no endpoint)'}: pricing unknown ({err})"
            total = None
        return (report, float(total) if total is not None else 0.0)


class FalSessionCosts:
    """Report every fal API call made this ComfyUI session and its total cost."""

    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("report", "total_usd")
    FUNCTION = "report"
    CATEGORY = _CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Report all fal API calls recorded this session (endpoints, request "
        "ids, estimated costs) and the running total in USD."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {},
            "optional": {
                "reset": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Clear the ledger after reporting",
                    },
                ),
                "trigger": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": (
                            "Connect any upstream text output (e.g. result_json) to "
                            "force this node to run after your generations finish"
                        ),
                    },
                ),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs: Any) -> Any:
        # The ledger mutates outside the graph; always re-run.
        return float("nan")

    def report(self, reset: bool = False, trigger: str = "") -> tuple[str, float]:
        del trigger  # Only used to order execution in the graph.
        ledger = SessionLedger()
        report_text = ledger.report()
        total = ledger.total_cost()
        if reset:
            ledger.reset()
        return (report_text, float(total))


def _output_directory() -> str:
    """ComfyUI's output directory, or ./output when running outside ComfyUI."""
    try:
        import folder_paths
    except ImportError:
        return os.path.abspath(os.path.join(os.getcwd(), "output"))
    return folder_paths.get_output_directory()


def _suffix_from_url(url: str) -> str:
    suffix = os.path.splitext(urlparse(url).path)[1]
    return suffix if suffix else ".bin"


def _resolve_save_directory(filename_prefix: str) -> tuple[str, str]:
    """Split the prefix into a confined save directory and a basename.

    The resolved directory must stay inside the ComfyUI output directory —
    a shared workflow must not be able to write outside it via '..' segments.
    """
    prefix = (filename_prefix or "").strip().strip("/") or "fal/media"
    subdir, basename = os.path.split(prefix)
    basename = basename or "media"
    output_root = os.path.realpath(_output_directory())
    directory = os.path.realpath(os.path.join(output_root, subdir))
    if directory != output_root and not directory.startswith(output_root + os.sep):
        raise FalApiError(
            "FalSaveMediaURL",
            f"filename_prefix escapes the output directory: {filename_prefix!r}",
        )
    return directory, basename


def _claim_unique_destination(directory: str, basename: str, suffix: str) -> str:
    """Atomically claim the first free '<basename>_00001<suffix>' path.

    O_CREAT|O_EXCL closes the check-then-act race between concurrent saves.
    """
    for counter in range(1, _SAVE_COUNTER_LIMIT):
        candidate = os.path.join(directory, f"{basename}_{counter:05d}{suffix}")
        try:
            os.close(os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            return candidate
        except FileExistsError:
            continue
    raise FalApiError(
        "FalSaveMediaURL",
        f"Could not find a free filename for '{basename}{suffix}' in {directory}",
    )


class FalSaveMediaURL:
    """Download a media URL and save it into the ComfyUI output directory."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "save"
    CATEGORY = _CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Download a result URL (video, audio, file, ...) and store it under "
        "the ComfyUI output directory with a unique, never-overwriting name."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "url": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "http(s) URL of the media to download and save",
                    },
                ),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "fal/media",
                        "tooltip": "Relative to the ComfyUI output directory; subfolders allowed",
                    },
                ),
            },
        }

    def save(self, url: str, filename_prefix: str = "fal/media") -> tuple[str]:
        target_url = (url or "").strip()
        if not target_url.startswith(("http://", "https://")):
            raise FalApiError(
                "FalSaveMediaURL",
                f"Expected an http(s) URL to save, got: {target_url!r}",
            )

        directory, basename = _resolve_save_directory(filename_prefix)
        suffix = _suffix_from_url(target_url)

        temp_path = MediaUtils.download_url_to_temp(target_url, suffix)
        try:
            os.makedirs(directory, exist_ok=True)
            destination = _claim_unique_destination(directory, basename, suffix)
            shutil.move(temp_path, destination)
        except FalApiError:
            raise
        except Exception as err:
            logger.error("FalSaveMediaURL: failed to save %s: %s", target_url, err)
            raise FalApiError(
                "FalSaveMediaURL", f"Failed to save {target_url}: {err}"
            ) from err
        finally:
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

        saved = os.path.abspath(destination)
        logger.info("FalSaveMediaURL: saved %s -> %s", target_url, saved)
        return (saved,)


NODE_CLASS_MAPPINGS = {
    "FalSubmit_fal": FalSubmit,
    "FalCollect_fal": FalCollect,
    "FalResultByRequestId_fal": FalResultByRequestId,
    "FalCostEstimator_fal": FalCostEstimator,
    "FalSessionCosts_fal": FalSessionCosts,
    "FalSaveMediaURL_fal": FalSaveMediaURL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalSubmit_fal": "Fal Submit (async) (fal)",
    "FalCollect_fal": "Fal Collect (async result) (fal)",
    "FalResultByRequestId_fal": "Fal Result by Request ID (fal)",
    "FalCostEstimator_fal": "Fal Cost Estimator (fal)",
    "FalSessionCosts_fal": "Fal Session Costs (fal)",
    "FalSaveMediaURL_fal": "Fal Save Media from URL (fal)",
}
