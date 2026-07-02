"""Self-test for the dynamic node package. Stdlib only; stubs the utils facade.

Run: python3 nodes/dynamic/_selftest.py
"""

from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
NODES_DIR = PACKAGE_DIR.parent
REPO_ROOT = NODES_DIR.parent
REAL_REGISTRY = REPO_ROOT / "data" / "fal_registry.json"

PKG = "falapi_nodes"


def _install_stub_facade() -> types.ModuleType:
    """Install a stub falapi_nodes.fal_utils satisfying the facade contract."""
    stub = types.ModuleType(f"{PKG}.fal_utils")

    class FalApiError(Exception):
        def __init__(self, endpoint, message):
            super().__init__(f"[{endpoint}] {message}")
            self.endpoint = endpoint
            self.message = message

    class FalConfig:
        def get_setting(self, section, name, default=None):
            return default

    class ImageUtils:
        @staticmethod
        def upload_image(tensor):
            return "https://stub.fal.media/image.png"

        @staticmethod
        def prepare_images(images):
            return ["https://stub.fal.media/1.png", "https://stub.fal.media/2.png"]

    class ResultProcessor:
        @staticmethod
        def process_image_result(result):
            return ("IMAGE_TENSOR",)

        @staticmethod
        def process_single_image_result(result):
            return ("IMAGE_TENSOR",)

    class ApiHandler:
        last_call = None

        @staticmethod
        def submit_and_get_result(endpoint, arguments):
            ApiHandler.last_call = (endpoint, arguments)
            return _CANNED_RESULTS.get(endpoint, {"ok": True})

    class MediaUtils:
        @staticmethod
        def video_from_url(url):
            return "VIDEO_OBJ"

        @staticmethod
        def audio_from_url(url):
            return {"waveform": None, "sample_rate": 44100}

        @staticmethod
        def upload_video(video):
            return "https://stub.fal.media/video.mp4"

        @staticmethod
        def upload_audio(audio):
            return "https://stub.fal.media/audio.wav"

        @staticmethod
        def download_url_to_temp(url, suffix):
            return "/tmp/stub" + suffix

    stub.FalApiError = FalApiError
    stub.FalConfig = FalConfig
    stub.ImageUtils = ImageUtils
    stub.ResultProcessor = ResultProcessor
    stub.ApiHandler = ApiHandler
    stub.MediaUtils = MediaUtils
    stub.logger = logging.getLogger("fal_stub")
    sys.modules[stub.__name__] = stub
    return stub


_CANNED_RESULTS = {
    "fal-ai/flux/dev": {"images": [{"url": "https://x/i.png"}], "seed": 1},
    "fal-ai/kling-video/v2/master/image-to-video": {
        "video": {"url": "https://x/v.mp4"}
    },
    "fal-ai/video-upscaler": {"video": {"url": "https://x/up.mp4"}},
    "fal-ai/kokoro/american-english": {"audio": {"url": "https://x/a.wav"}},
    "tripo3d/tripo/v2.5/image-to-3d": {"model_mesh": {"url": "https://x/m.glb"}},
}


def _install_package() -> None:
    pkg = types.ModuleType(PKG)
    pkg.__path__ = [str(NODES_DIR)]
    sys.modules[PKG] = pkg


def _load_models(path: Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["models"]


def _check_registry(models, factory, outputs, label):
    keys = set()
    names = set()
    skipped = {}
    built = 0
    for model in models:
        try:
            cls = factory.build_node_class(model)
            input_types = cls.INPUT_TYPES()
            assert isinstance(input_types, dict) and "required" in input_types
            assert "force_rerun" in input_types.get("optional", {})
            kind = model.get("output_kind", "json")
            expected = outputs.RETURN_SPECS.get(kind, outputs.RETURN_SPECS["json"])
            assert cls.RETURN_TYPES == expected[0], (
                f"RETURN_TYPES mismatch for {model['endpoint_id']}"
            )
            assert cls.RETURN_NAMES == expected[1]
            key = factory.node_key(model)
            assert key not in keys, f"key collision: {key}"
            keys.add(key)
            names.add(factory.build_display_name(model))
            built += 1
        except Exception as err:
            reason = type(err).__name__ + ": " + str(err)[:80]
            skipped[reason] = skipped.get(reason, 0) + 1
    print(f"[{label}] built={built} skipped={sum(skipped.values())}")
    if skipped:
        print(f"[{label}] skip reasons histogram:")
        for reason, count in sorted(skipped.items(), key=lambda kv: -kv[1]):
            print(f"    {count:4d}  {reason}")
    return built, skipped


def _test_fixture_behaviour(dyn, stub):
    from importlib import import_module

    arguments = import_module(f"{PKG}.dynamic.arguments")
    factory = import_module(f"{PKG}.dynamic.factory")
    import_module(f"{PKG}.dynamic.outputs")

    models = _load_models(PACKAGE_DIR / "_fixture_registry.json")
    by_id = {m["endpoint_id"]: m for m in models}

    # --- arguments: custom_size, seed=-1 omitted, empty json skipped ---
    flux = by_id["fal-ai/flux/dev"]
    kwargs = {
        "prompt": "a cat",
        "image_size": "custom_size",
        "width": 512,
        "height": 768,
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
        "seed": -1,
        "num_images": 1,
        "enable_safety_checker": True,
        "loras": "",
        "force_rerun": False,
    }
    kwargs_snapshot = dict(kwargs)
    args = arguments.build_arguments(flux, kwargs)
    assert args["image_size"] == {"width": 512, "height": 768}, args
    assert "seed" not in args and "loras" not in args and "force_rerun" not in args
    assert kwargs == kwargs_snapshot, "kwargs were mutated"

    # seed forwarded when != -1; enum passthrough
    args2 = arguments.build_arguments(flux, {**kwargs, "seed": 42, "image_size": "square"})
    assert args2["seed"] == 42 and args2["image_size"] == "square"

    # invalid json raises FalApiError
    try:
        arguments.build_arguments(flux, {**kwargs, "loras": "{not json"})
        raise AssertionError("expected FalApiError for bad JSON")
    except stub.FalApiError:
        pass

    # valid json parsed
    args3 = arguments.build_arguments(flux, {**kwargs, "loras": '[{"path": "x"}]'})
    assert args3["loras"] == [{"path": "x"}]

    # --- media uploads ---
    kling = by_id["fal-ai/kling-video/v2/master/image-to-video"]
    kargs = arguments.build_arguments(
        kling, {"prompt": "move", "image_url": "TENSOR", "duration": "5",
                "negative_prompt": "", "cfg_scale": 0.5}
    )
    assert kargs["image_url"] == "https://stub.fal.media/image.png"
    assert "negative_prompt" not in kargs  # optional empty string skipped

    tripo = by_id["tripo3d/tripo/v2.5/image-to-3d"]
    targs = arguments.build_arguments(
        tripo, {"image_url": "TENSOR", "image_urls": "BATCH", "texture": "HD", "seed": -1}
    )
    assert targs["image_urls"] == [
        "https://stub.fal.media/1.png",
        "https://stub.fal.media/2.png",
    ]

    upscaler = by_id["fal-ai/video-upscaler"]
    uargs = arguments.build_arguments(upscaler, {"video_url": "VIDEO_OBJ", "scale": 2.0})
    assert uargs["video_url"] == "https://stub.fal.media/video.mp4"

    # --- end-to-end run() per output kind ---
    flux_node = factory.build_node_class(flux)()
    assert flux_node.run(**kwargs) == ("IMAGE_TENSOR", "https://x/i.png")

    kling_node = factory.build_node_class(kling)()
    out = kling_node.run(prompt="move", image_url="TENSOR", duration="5",
                         negative_prompt="", cfg_scale=0.5)
    assert out == ("VIDEO_OBJ", "https://x/v.mp4"), out

    tts = by_id["fal-ai/kokoro/american-english"]
    tts_node = factory.build_node_class(tts)()
    audio_out = tts_node.run(prompt="hello", voice="af_heart", speed=1.0)
    assert audio_out[1] == "https://x/a.wav" and isinstance(audio_out[0], dict)

    tripo_node = factory.build_node_class(tripo)()
    file_out = tripo_node.run(image_url="TENSOR", texture="HD", seed=-1)
    assert file_out == ("https://x/m.glb",), file_out

    # --- IS_CHANGED semantics ---
    cls = factory.build_node_class(flux)
    h1 = cls.IS_CHANGED(prompt="a", force_rerun=False)
    h2 = cls.IS_CHANGED(prompt="a", force_rerun=False)
    h3 = cls.IS_CHANGED(prompt="b", force_rerun=False)
    nan = cls.IS_CHANGED(prompt="a", force_rerun=True)
    assert h1 == h2 and h1 != h3 and nan != nan  # nan != nan

    # --- loader end-to-end (fixture fallback path) ---
    classes, display = dyn.get_dynamic_mappings()
    assert "FalAnyEndpoint_fal" in classes
    assert len(classes) == len(display)
    assert len(set(display.values())) == len(display), "display name collision"
    if not REAL_REGISTRY.is_file():
        assert len(classes) == 6, f"expected 5 fixture + any-endpoint, got {len(classes)}"

    # --- any endpoint node ---
    any_cls = classes["FalAnyEndpoint_fal"]
    node = any_cls()
    any_cls.INPUT_TYPES()
    res = node.run(
        endpoint_id="fal-ai/flux/dev",
        arguments_json='{"prompt": "hi", "image_url": "should-be-overridden"}',
        image="TENSOR",
        image_2="TENSOR2",
        seed=7,
    )
    endpoint, sent = sys.modules[f"{PKG}.fal_utils"].ApiHandler.last_call
    assert sent["image_url"] == "https://stub.fal.media/image.png"  # media wins
    assert sent["image_urls"] == [
        "https://stub.fal.media/image.png",
        "https://stub.fal.media/image.png",
    ]
    assert sent["seed"] == 7 and sent["prompt"] == "hi"
    assert res[0] == "IMAGE_TENSOR" and json.loads(res[3])["seed"] == 1

    print("[fixture] behaviour tests passed")


def _file_input(name, media_kind, required=False, is_list=False, type_="string"):
    return {
        "name": name, "type": type_, "required": required, "default": None,
        "enum": None, "min": None, "max": None, "description": "",
        "media_kind": media_kind, "is_list": is_list, "multiline": False,
        "has_custom_size": False,
    }


def _test_direct_url_passthrough(stub):
    from importlib import import_module

    arguments = import_module(f"{PKG}.dynamic.arguments")
    factory = import_module(f"{PKG}.dynamic.factory")
    schema = import_module(f"{PKG}.dynamic.schema_to_inputs")

    models = _load_models(PACKAGE_DIR / "_fixture_registry.json")
    by_id = {m["endpoint_id"]: m for m in models}
    kling = by_id["fal-ai/kling-video/v2/master/image-to-video"]
    tripo = by_id["tripo3d/tripo/v2.5/image-to-3d"]
    flux = by_id["fal-ai/flux/dev"]

    # --- twins present and placed: required media → start of optional ---
    it = schema.build_input_types(kling)
    assert list(it["optional"])[0] == "image_url_direct_url"
    assert it["optional"]["image_url_direct_url"][0] == "STRING"
    assert "image_url_direct_url" not in it["required"]

    uit = schema.build_input_types(by_id["fal-ai/video-upscaler"])
    assert list(uit["optional"])[0] == "video_url_direct_url"

    # optional is_list media → twin immediately after its media input
    tit = schema.build_input_types(tripo)
    tkeys = list(tit["optional"])
    assert tkeys[0] == "image_url_direct_url"
    assert tkeys.index("image_urls_direct_url") == tkeys.index("image_urls") + 1
    assert "Comma" in tit["optional"]["image_urls_direct_url"][1]["tooltip"]

    # no twin for text-only models or media_kind "file"
    fit = schema.build_input_types(flux)
    assert not any(k.endswith("_direct_url") for k in {**fit["required"], **fit["optional"]})
    file_model = {**kling, "inputs": [_file_input("doc_url", "file", required=True)]}
    ffit = schema.build_input_types(file_model)
    assert not any(k.endswith("_direct_url") for k in {**ffit["required"], **ffit["optional"]})

    # collision guard: a literal *_direct_url input suppresses the generated twin
    collide = {**kling, "inputs": [
        _file_input("image_url", "image", required=True),
        _file_input("image_url_direct_url", None),
    ]}
    cit = schema.build_input_types(collide)
    assert list(cit["optional"]).count("image_url_direct_url") == 1

    # --- passthrough beats tensor upload; twin key never leaks ---
    kargs = arguments.build_arguments(kling, {
        "prompt": "move", "image_url": "TENSOR",
        "image_url_direct_url": "  https://cdn.fal.media/start.png  ",
        "duration": "5", "negative_prompt": "", "cfg_scale": 0.5,
    })
    assert kargs["image_url"] == "https://cdn.fal.media/start.png"
    assert not any(k.endswith("_direct_url") for k in kargs)

    # media input None or absent: URL still wins
    for image_value in ({"image_url": None}, {}):
        k2 = arguments.build_arguments(
            kling,
            {"prompt": "m", "image_url_direct_url": "https://cdn.fal.media/s.png", **image_value},
        )
        assert k2["image_url"] == "https://cdn.fal.media/s.png"

    # blank twin falls back to the normal upload path
    k3 = arguments.build_arguments(
        kling, {"prompt": "m", "image_url": "TENSOR", "image_url_direct_url": "   "}
    )
    assert k3["image_url"] == "https://stub.fal.media/image.png"

    # non-http(s) raises
    try:
        arguments.build_arguments(kling, {"prompt": "x", "image_url_direct_url": "ftp://nope"})
        raise AssertionError("expected FalApiError for non-http(s) direct URL")
    except stub.FalApiError:
        pass

    # is_list twin: comma-separated string → list of URLs
    targs = arguments.build_arguments(tripo, {
        "image_url": "TENSOR", "texture": "HD", "seed": -1,
        "image_urls_direct_url": "https://a/1.png, https://a/2.png ,https://a/3.png",
    })
    assert targs["image_urls"] == ["https://a/1.png", "https://a/2.png", "https://a/3.png"]
    assert targs["image_url"] == "https://stub.fal.media/image.png"
    assert not any(k.endswith("_direct_url") for k in targs)

    try:
        arguments.build_arguments(tripo, {"image_urls_direct_url": "https://a/1.png, nope"})
        raise AssertionError("expected FalApiError for bad URL in list")
    except stub.FalApiError:
        pass

    # collision model: the literal input passes through as a plain string argument
    cargs = arguments.build_arguments(
        collide, {"image_url": "TENSOR", "image_url_direct_url": "not-a-url"}
    )
    assert cargs["image_url"] == "https://stub.fal.media/image.png"
    assert cargs["image_url_direct_url"] == "not-a-url"

    # --- skip_cache plumbing: old signature tolerated, new one receives the flag ---
    node = factory.build_node_class(flux)()
    node.run(prompt="hi", force_rerun=True)
    assert len(stub.ApiHandler.last_call) == 2  # legacy stub: called without skip_cache

    def with_skip(endpoint, arguments, timeout=None, skip_cache=False):
        stub.ApiHandler.last_call = (endpoint, arguments, skip_cache)
        return _CANNED_RESULTS.get(endpoint, {"ok": True})

    original = stub.ApiHandler.submit_and_get_result
    stub.ApiHandler.submit_and_get_result = staticmethod(with_skip)
    try:
        node.run(prompt="hi", force_rerun=True)
        assert stub.ApiHandler.last_call[2] is True
        node.run(prompt="hi", force_rerun=False)
        assert stub.ApiHandler.last_call[2] is False
    finally:
        stub.ApiHandler.submit_and_get_result = original

    print("[fixture] direct-url passthrough tests passed")


def _twin_sweep(models, schema):
    """Real-registry sweep: build every INPUT_TYPES, count twin coverage."""
    gained_nodes = 0
    twin_count = 0
    for model in models:
        input_types = schema.build_input_types(model)
        names = {inp["name"] for inp in model.get("inputs", [])}
        overlap = set(input_types["required"]) & set(input_types["optional"])
        assert not overlap, f"{model['endpoint_id']}: bucket overlap {overlap}"
        twins = [
            key for key in input_types["optional"]
            if key.endswith("_direct_url") and key not in names
        ]
        if twins:
            gained_nodes += 1
            twin_count += len(twins)
    print(f"[real] direct-url twins: {twin_count} twin inputs across "
          f"{gained_nodes}/{len(models)} nodes")


def _dump_samples(models, factory):
    samples = [
        ("flux", "text-to-image"),
        ("kling", "image-to-video"),
        (None, "text-to-speech"),
        (None, "image-to-3d"),
        (None, "video-to-video"),
    ]
    seen = set()
    for hint, category in samples:
        candidates = [
            m for m in models
            if m.get("category") == category and m["endpoint_id"] not in seen
        ]
        model = next(
            (m for m in candidates if hint and hint in m["endpoint_id"]),
            candidates[0] if candidates else None,
        )
        if model is None:
            print(f"-- no sample for {hint or category}")
            continue
        seen.add(model["endpoint_id"])
        cls = factory.build_node_class(model)
        print(f"\n-- INPUT_TYPES for {model['endpoint_id']} "
              f"({model.get('output_kind')}):")
        print(json.dumps(cls.INPUT_TYPES(), indent=2, default=str)[:2500])


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    _install_package()
    stub = _install_stub_facade()

    from importlib import import_module

    dyn = import_module(f"{PKG}.dynamic")
    factory = import_module(f"{PKG}.dynamic.factory")
    outputs = import_module(f"{PKG}.dynamic.outputs")

    fixture_models = _load_models(PACKAGE_DIR / "_fixture_registry.json")
    built, _ = _check_registry(fixture_models, factory, outputs, "fixture")
    assert built == 5

    _test_fixture_behaviour(dyn, stub)
    _test_direct_url_passthrough(stub)

    if REAL_REGISTRY.is_file():
        schema = import_module(f"{PKG}.dynamic.schema_to_inputs")
        real_models = _load_models(REAL_REGISTRY)
        built, skipped = _check_registry(real_models, factory, outputs, "real")
        _twin_sweep(real_models, schema)
        classes, display = dyn.get_dynamic_mappings()
        print(f"[real] loader registered {len(classes)} nodes "
              f"(incl. any-endpoint), display names unique: "
              f"{len(set(display.values())) == len(display)}")
        _dump_samples(real_models, factory)
    else:
        print("[real] data/fal_registry.json not present; skipped real-registry checks")

    print("\nSELFTEST OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
