"""Tests for the FAL/Utils node layer (dataset, image, data, video basics)."""

from __future__ import annotations

import importlib
import json
import zipfile

import pytest
import torch
from conftest import PKG, _load_package


@pytest.fixture(scope="session")
def archive_mod():
    _load_package()
    return importlib.import_module(f"{PKG}.nodes.utils.archive")


@pytest.fixture(scope="session")
def image_nodes(pack):
    return pack.NODE_CLASS_MAPPINGS


def test_zip_images_with_captions(archive_mod, tmp_path):
    images = torch.rand(3, 8, 8, 3)
    zip_path = archive_mod.ArchiveUtils.zip_images(images, captions=["a", "", "c"])
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = sorted(zf.namelist())
            assert "image_0.png" in names and "image_2.txt" in names
            assert zf.read("image_0.txt").decode() == "a"
    finally:
        import os

        os.unlink(zip_path)


def test_zip_images_caption_mismatch_raises(archive_mod, errors_mod):
    with pytest.raises(errors_mod.FalApiError):
        archive_mod.ArchiveUtils.zip_images(torch.rand(2, 8, 8, 3), captions=["only one"])


def test_json_extract(pack):
    cls = pack.NODE_CLASS_MAPPINGS["FalJSONExtract_fal"]
    node = cls()
    fn = getattr(node, cls.FUNCTION)
    payload = json.dumps({"video": {"url": "https://x/v.mp4"}, "images": [{"url": "https://x/i.png"}], "seed": 42})
    assert fn(json_text=payload, path="video.url", default="")[0] == "https://x/v.mp4"
    assert fn(json_text=payload, path="images[0].url", default="")[0] == "https://x/i.png"
    assert fn(json_text=payload, path="seed", default="")[1] == 42.0
    assert fn(json_text=payload, path="missing.path", default="fallback")[0] == "fallback"


def test_prompt_lines_wraps(pack):
    cls = pack.NODE_CLASS_MAPPINGS["FalPromptLines_fal"]
    node = cls()
    fn = getattr(node, cls.FUNCTION)
    text = "one\ntwo\nthree"
    assert fn(text=text, index=0, skip_blank=True)[0] == "one"
    assert fn(text=text, index=4, skip_blank=True)[0] == "two"  # wraps modulo 3


def test_resize_to_preset_dims(pack):
    cls = pack.NODE_CLASS_MAPPINGS["FalResizeToPreset_fal"]
    node = cls()
    fn = getattr(node, cls.FUNCTION)
    image = torch.rand(1, 300, 500, 3)
    out, width, height = fn(image=image, preset="landscape_16_9", width=1024, height=1024, mode="cover_crop")
    assert (width, height) == (1024, 576)
    assert tuple(out.shape) == (1, 576, 1024, 3)


def test_base64_round_trip(pack):
    cm = pack.NODE_CLASS_MAPPINGS
    enc_cls, dec_cls = cm["FalImageToBase64_fal"], cm["FalBase64ToImage_fal"]
    image = torch.rand(1, 16, 16, 3)
    encoded = getattr(enc_cls(), enc_cls.FUNCTION)(image=image, format="png", data_uri=True)[0]
    decoded = getattr(dec_cls(), dec_cls.FUNCTION)(data=encoded)[0]
    assert tuple(decoded.shape) == (1, 16, 16, 3)
    assert torch.allclose(image, decoded, atol=2 / 255)


def test_image_grid_shape(pack):
    cls = pack.NODE_CLASS_MAPPINGS["FalImageGrid_fal"]
    node = cls()
    fn = getattr(node, cls.FUNCTION)
    out = fn(images=torch.rand(4, 32, 32, 3), labels="a\nb\nc\nd", columns=2, cell_padding=4, label_height=16)[0]
    assert out.ndim == 4 and out.shape[0] == 1 and out.shape[3] == 3


def test_extract_frames_from_real_video(pack, tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    path = str(tmp_path / "clip.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 8, (32, 32))
    for i in range(16):
        frame = np.full((32, 32, 3), 255 if i == 15 else 0, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    cls = pack.NODE_CLASS_MAPPINGS["FalExtractFrames_fal"]
    node = cls()
    fn = getattr(node, cls.FUNCTION)
    frames, count = fn(video=path, mode="last", n=1, max_frames=64)
    assert count == 16
    assert frames.shape[0] == 1
    assert frames.mean().item() > 0.9  # last frame is white


def test_all_util_nodes_have_tooltips(pack):
    util_keys = [k for k, c in pack.NODE_CLASS_MAPPINGS.items() if c.CATEGORY.startswith("FAL/Utils")]
    assert len(util_keys) == 20
    for key in util_keys:
        input_types = pack.NODE_CLASS_MAPPINGS[key].INPUT_TYPES()
        for bucket in ("required", "optional"):
            for name, spec in input_types.get(bucket, {}).items():
                if len(spec) > 1 and isinstance(spec[1], dict):
                    assert "tooltip" in spec[1], f"{key}.{name} missing tooltip"
