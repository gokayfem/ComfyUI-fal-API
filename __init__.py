import importlib
import importlib.util

node_list = [
    "image_node",
    "video_node",
    "llm_node",
    "vlm_node",
    "trainer_node",
    "upscaler_node",
    "platform_node",
    "billing_node",
    "inbox_node",
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for module_name in node_list:
    imported_module = importlib.import_module(f".nodes.{module_name}", __name__)

    NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **imported_module.NODE_CLASS_MAPPINGS}
    NODE_DISPLAY_NAME_MAPPINGS = {
        **NODE_DISPLAY_NAME_MAPPINGS,
        **imported_module.NODE_DISPLAY_NAME_MAPPINGS,
    }

try:
    from .nodes.dynamic import get_dynamic_mappings

    dyn_classes, dyn_display = get_dynamic_mappings()
    # static nodes win on any key collision
    for k, v in dyn_classes.items():
        NODE_CLASS_MAPPINGS.setdefault(k, v)
    for k, v in dyn_display.items():
        NODE_DISPLAY_NAME_MAPPINGS.setdefault(k, v)
except Exception as _dynamic_error:  # never break static nodes
    import logging

    logging.getLogger(__name__).error(
        "Failed to load dynamic fal nodes: %s", _dynamic_error
    )


WEB_DIRECTORY = "./web"

try:
    from .nodes import server_routes as _server_routes  # noqa: F401  registers /fal_api routes
except Exception as _routes_error:  # never break node loading over HTTP extras
    import logging

    logging.getLogger(__name__).warning(
        "fal API server routes not registered: %s", _routes_error
    )

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
