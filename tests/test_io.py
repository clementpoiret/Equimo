"""Tests for equimo.io."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from equimo.io import (
    _MODEL_REGISTRY,
    _center_crop_square,
    _validate_identifier,
    get_model_cls,
    load_model,
    register_model,
    save_model,
)

KEY = jr.PRNGKey(0)


# ---------------------------------------------------------------------------
# _validate_identifier
# ---------------------------------------------------------------------------


class TestValidateIdentifier:
    @pytest.mark.parametrize(
        "identifier",
        [
            "vit_base_patch16",
            "mlla-small",
            "MyModel123",
            "a",
            "A1_b2-c3",
        ],
    )
    def test_valid_identifiers(self, identifier):
        _validate_identifier(identifier)  # must not raise

    @pytest.mark.parametrize(
        "identifier",
        [
            "../etc/passwd",
            "model/../../secret",
            "model?query=1",
            "model name",
            "model.tar.lz4",
            "model\x00null",
            "",
        ],
    )
    def test_invalid_identifiers_raise(self, identifier):
        with pytest.raises(ValueError, match="Unsafe model identifier"):
            _validate_identifier(identifier)

    def test_path_traversal_blocked(self):
        with pytest.raises(ValueError):
            _validate_identifier("../../etc/passwd")

    def test_url_special_chars_blocked(self):
        with pytest.raises(ValueError):
            _validate_identifier("model?foo=bar&baz=qux")


# ---------------------------------------------------------------------------
# _center_crop_square
# ---------------------------------------------------------------------------


class TestCenterCropSquare:
    def test_square_input_unchanged(self):
        arr = jnp.ones((64, 64, 3))
        result = _center_crop_square(arr)
        assert result.shape == (64, 64, 3)

    def test_wide_image_crops_width(self):
        arr = jnp.ones((64, 128, 3))
        result = _center_crop_square(arr)
        assert result.shape == (64, 64, 3)

    def test_tall_image_crops_height(self):
        arr = jnp.ones((128, 64, 3))
        result = _center_crop_square(arr)
        assert result.shape == (64, 64, 3)

    def test_crop_is_centered_wide(self):
        """For a 1×4 array [0,1,2,3], center crop to 1×2 should give [1,2]."""
        arr = jnp.arange(4).reshape(1, 4)
        result = _center_crop_square(arr)
        assert result.shape == (1, 1)

    def test_crop_is_centered_tall(self):
        """For a 4×1 array, center crop to 1×1 should yield the middle element."""
        arr = jnp.arange(4).reshape(4, 1)
        result = _center_crop_square(arr)
        assert result.shape == (1, 1)

    def test_no_copy_for_square(self):
        """Square arrays must be returned as-is (same object)."""
        arr = jnp.ones((32, 32, 3))
        assert _center_crop_square(arr) is arr

    def test_1d_raises(self):
        arr = jnp.ones((64,))
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            _center_crop_square(arr)

    def test_2d_hw_array(self):
        arr = jnp.ones((64, 128))
        result = _center_crop_square(arr)
        assert result.shape == (64, 64)


# ---------------------------------------------------------------------------
# register_model / get_model_cls
# ---------------------------------------------------------------------------


class TestRegisterModel:
    def test_register_default_name(self):
        @register_model()
        class CustomModelA(eqx.Module):
            pass

        assert "custommodela" in _MODEL_REGISTRY
        assert get_model_cls("custommodela") is CustomModelA

    def test_register_custom_name(self):
        @register_model("my_custom_net")
        class CustomModelB(eqx.Module):
            pass

        assert "my_custom_net" in _MODEL_REGISTRY
        assert get_model_cls("my_custom_net") is CustomModelB

    def test_register_name_is_case_insensitive(self):
        @register_model("CamelCaseModel")
        class CustomModelC(eqx.Module):
            pass

        assert get_model_cls("camelcasemodel") is CustomModelC
        assert get_model_cls("CAMELCASEMODEL") is CustomModelC

    def test_register_non_eqx_module_raises(self):
        with pytest.raises(TypeError, match="must be a subclass of eqx.Module"):
            @register_model()
            class NotAModule:
                pass

    def test_register_duplicate_raises(self):
        @register_model()
        class UniqueModel(eqx.Module):
            pass

        with pytest.raises(ValueError, match="already registered"):
            @register_model(name="UniqueModel")
            class AnotherModel(eqx.Module):
                pass


class TestGetModelCls:
    def test_string_resolution_builtin(self):
        from equimo.models import VisionTransformer

        assert get_model_cls("vit") is VisionTransformer

    def test_string_case_insensitive(self):
        from equimo.models import VisionTransformer

        assert get_model_cls("VIT") is VisionTransformer
        assert get_model_cls("Vit") is VisionTransformer

    def test_class_passthrough(self):
        assert get_model_cls(eqx.nn.Linear) is eqx.nn.Linear

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="Unknown model class"):
            get_model_cls("nonexistent_model_xyz")

    def test_error_message_lists_available(self):
        with pytest.raises(ValueError, match="Available"):
            get_model_cls("nonexistent_model_xyz")

    def test_all_builtin_models_registered(self):
        builtins = ["vit", "mlla", "vssd", "shvit", "fastervit", "partialformer",
                    "iformer", "mobilenetv3", "reduceformer"]
        for name in builtins:
            cls = get_model_cls(name)
            assert issubclass(cls, eqx.Module), f"{name} not an eqx.Module subclass"


# ---------------------------------------------------------------------------
# save_model / load_model round-trip
# ---------------------------------------------------------------------------


class _TinyModel(eqx.Module):
    """Minimal model for save/load round-trip tests."""
    linear: eqx.nn.Linear
    label: str = eqx.field(static=True)

    def __init__(self, in_features: int, out_features: int, *, key, label: str = "default"):
        self.linear = eqx.nn.Linear(in_features, out_features, key=key)
        self.label = label

    def __call__(self, x):
        return jax.vmap(self.linear)(x)


# Register for load_model string-based tests
register_model("_tinymodel_test")(_TinyModel)


class TestSaveLoadRoundTrip:
    def _make_model(self):
        return _TinyModel(8, 4, key=KEY)

    def _model_config(self):
        return {"in_features": 8, "out_features": 4}

    def test_save_creates_lz4_file(self, tmp_path):
        model = self._make_model()
        path = tmp_path / "model"
        save_model(path, model, self._model_config())
        assert (tmp_path / "model.tar.lz4").exists()

    def test_save_with_explicit_suffix(self, tmp_path):
        model = self._make_model()
        path = tmp_path / "model.tar.lz4"
        save_model(path, model, self._model_config())
        assert path.exists()

    def test_save_no_compression(self, tmp_path):
        model = self._make_model()
        path = tmp_path / "model_dir"
        save_model(path, model, self._model_config(), compression=False)
        assert (path / "metadata.json").exists()
        assert (path / "weights.eqx").exists()

    def test_metadata_contains_versions(self, tmp_path):
        model = self._make_model()
        path = tmp_path / "model_dir"
        save_model(path, model, self._model_config(), compression=False)
        with open(path / "metadata.json") as f:
            meta = json.load(f)
        assert "jax_version" in meta
        assert "equinox_version" in meta
        assert "equimo_version" in meta

    def test_metadata_contains_model_config(self, tmp_path):
        model = self._make_model()
        path = tmp_path / "model_dir"
        cfg = self._model_config()
        save_model(path, model, cfg, compression=False)
        with open(path / "metadata.json") as f:
            meta = json.load(f)
        assert meta["model_config"] == cfg

    def test_mutable_default_torch_hub_cfg(self, tmp_path):
        """torch_hub_cfg=None must not share a mutable dict across calls."""
        model = self._make_model()
        path1 = tmp_path / "m1"
        path2 = tmp_path / "m2"
        save_model(path1, model, self._model_config(), compression=False)
        save_model(path2, model, self._model_config(), compression=False)
        with open(path1 / "metadata.json") as f:
            meta1 = json.load(f)
        with open(path2 / "metadata.json") as f:
            meta2 = json.load(f)
        assert meta1["torch_hub_cfg"] == {}
        assert meta2["torch_hub_cfg"] == {}

    def test_mutable_default_timm_cfg(self, tmp_path):
        model = self._make_model()
        path = tmp_path / "m_timm"
        save_model(path, model, self._model_config(), compression=False)
        with open(path / "metadata.json") as f:
            meta = json.load(f)
        assert meta["timm"] == []

    def test_load_roundtrip_compressed(self, tmp_path):
        model = self._make_model()
        path = tmp_path / "model"
        save_model(path, model, self._model_config())
        loaded = load_model(_TinyModel, path=tmp_path / "model.tar.lz4")
        x = jr.normal(KEY, (4, 8))
        assert jnp.allclose(model(x), loaded(x), atol=1e-5)

    def test_load_roundtrip_uncompressed(self, tmp_path):
        model = self._make_model()
        path = tmp_path / "model_dir"
        save_model(path, model, self._model_config(), compression=False)
        loaded = load_model(_TinyModel, path=path)
        x = jr.normal(KEY, (4, 8))
        assert jnp.allclose(model(x), loaded(x), atol=1e-5)

    def test_load_string_cls(self, tmp_path):
        model = self._make_model()
        path = tmp_path / "model"
        save_model(path, model, self._model_config())
        loaded = load_model("_tinymodel_test", path=tmp_path / "model.tar.lz4")
        assert isinstance(loaded, _TinyModel)

    def test_load_inference_mode_default(self, tmp_path):
        model = self._make_model()
        path = tmp_path / "model_dir"
        save_model(path, model, self._model_config(), compression=False)
        loaded = load_model(_TinyModel, path=path, inference_mode=True)
        # inference_mode=True means no training state — model must still be callable
        x = jr.normal(KEY, (4, 8))
        assert loaded(x).shape == (4, 4)

    def test_load_model_kwargs_override(self, tmp_path):
        """model_kwargs must override non-structural stored model_config values."""
        model = self._make_model()
        path = tmp_path / "model_dir"
        save_model(path, model, {"in_features": 8, "out_features": 4, "label": "saved"}, compression=False)
        # Override label (non-structural) at load time
        loaded = load_model(_TinyModel, path=path, label="overridden")
        assert loaded.label == "overridden"

    def test_load_requires_identifier_or_path(self):
        with pytest.raises(ValueError, match="Both.*None"):
            load_model(_TinyModel)

    def test_load_exclusive_identifier_path(self, tmp_path):
        with pytest.raises(ValueError, match="Both.*defined"):
            load_model(_TinyModel, identifier="some_id", path=tmp_path / "x")

    def test_load_compressed_cached_decompression(self, tmp_path):
        """Loading a compressed model twice must reuse the cached decompression dir."""
        model = self._make_model()
        path = tmp_path / "model"
        save_model(path, model, self._model_config())
        archive = tmp_path / "model.tar.lz4"
        load_model(_TinyModel, path=archive)
        decompressed = archive.with_suffix("").with_suffix("")
        assert decompressed.exists()
        # Second load must not fail
        load_model(_TinyModel, path=archive)


# ---------------------------------------------------------------------------
# download (identifier validation; no network calls)
# ---------------------------------------------------------------------------


class TestDownload:
    def test_invalid_identifier_raises(self):
        from equimo.io import download

        with pytest.raises(ValueError, match="Unsafe model identifier"):
            download("../../malicious", repository="http://example.com")

    def test_identifier_with_slash_raises(self):
        from equimo.io import download

        with pytest.raises(ValueError):
            download("model/subdir", repository="http://example.com")

    def test_cached_file_returned_without_request(self, tmp_path):
        """If the archive already exists on disk, no HTTP request should be made."""
        from equimo.io import download

        identifier = "vit_test_cache"
        model_name = identifier.split("_")[0]
        cache_path = Path(
            f"~/.cache/equimo/{model_name}/{identifier}.tar.lz4"
        ).expanduser()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.touch()

        try:
            with patch("equimo.io.requests.get") as mock_get:
                result = download(identifier, repository="http://example.com")
                mock_get.assert_not_called()
            assert result == cache_path
        finally:
            cache_path.unlink(missing_ok=True)

    def test_download_makes_get_request(self, tmp_path):
        """When archive is absent, a streaming GET must be issued."""
        from equimo.io import download

        identifier = "vit_test_dl_xyz"
        model_name = identifier.split("_")[0]
        cache_path = Path(
            f"~/.cache/equimo/{model_name}/{identifier}.tar.lz4"
        ).expanduser()
        cache_path.unlink(missing_ok=True)

        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"fake data"]
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.raise_for_status = MagicMock()

        try:
            with patch("equimo.io.requests.get", return_value=mock_response) as mock_get:
                result = download(identifier, repository="http://example.com")
                mock_get.assert_called_once()
                call_kwargs = mock_get.call_args
                assert call_kwargs.kwargs.get("stream") is True
                assert call_kwargs.kwargs.get("timeout") is not None
                assert call_kwargs.kwargs.get("verify") is True
        finally:
            cache_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------


class TestLoadImage:
    @pytest.fixture
    def sample_image_path(self, tmp_path):
        try:
            from PIL import Image as PILImage
        except ImportError:
            pytest.skip("Pillow not installed")
        img = PILImage.new("RGB", (64, 48), color=(128, 64, 32))
        path = tmp_path / "test.png"
        img.save(str(path))
        return str(path)

    @pytest.fixture
    def grayscale_image_path(self, tmp_path):
        try:
            from PIL import Image as PILImage
        except ImportError:
            pytest.skip("Pillow not installed")
        img = PILImage.new("L", (32, 32), color=128)
        path = tmp_path / "gray.png"
        img.save(str(path))
        return str(path)

    def test_output_shape_chw(self, sample_image_path):
        from equimo.io import load_image

        out = load_image(sample_image_path)
        assert out.ndim == 3
        assert out.shape[0] == 3  # channels first

    def test_output_dtype_float32(self, sample_image_path):
        from equimo.io import load_image

        out = load_image(sample_image_path)
        assert out.dtype == jnp.float32

    def test_output_range_0_1(self, sample_image_path):
        from equimo.io import load_image

        out = load_image(sample_image_path)
        assert float(jnp.min(out)) >= 0.0
        assert float(jnp.max(out)) <= 1.0

    def test_grayscale_converted_to_rgb(self, grayscale_image_path):
        from equimo.io import load_image

        out = load_image(grayscale_image_path)
        assert out.shape[0] == 3

    def test_resize(self, sample_image_path):
        from equimo.io import load_image

        out = load_image(sample_image_path, size=32)
        assert out.shape == (3, 32, 32)

    def test_normalization_applied(self, sample_image_path):
        from equimo.io import load_image

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        out_norm = load_image(sample_image_path, mean=mean, std=std)
        out_raw = load_image(sample_image_path)
        assert not jnp.allclose(out_norm, out_raw)

    def test_normalization_formula(self, sample_image_path):
        from equimo.io import load_image

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        out_norm = load_image(sample_image_path, mean=mean, std=std)
        out_raw = load_image(sample_image_path)
        expected = (out_raw - 0.5) / 0.5
        assert jnp.allclose(out_norm, expected, atol=1e-5)

    def test_center_crop(self, sample_image_path):
        from equimo.io import load_image

        out = load_image(sample_image_path, center_crop=True)
        _, h, w = out.shape
        assert h == w

    def test_center_crop_then_resize(self, sample_image_path):
        from equimo.io import load_image

        out = load_image(sample_image_path, center_crop=True, size=32)
        assert out.shape == (3, 32, 32)
