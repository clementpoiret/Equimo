import io
import json
import os
import re
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import lz4.frame
import requests
from jaxtyping import Array, Float
from loguru import logger
from semver import Version

import equimo.models as em
from equimo import __version__

DEFAULT_REPOSITORY_URL = (
    "https://huggingface.co/poiretclement/equimo/resolve/main/models/default"
)

# Identifier must only contain alphanumerics, hyphens, and underscores.
# This prevents path traversal when identifiers are embedded in local file paths.
_SAFE_IDENTIFIER_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")
_MODEL_REGISTRY: dict[str, type[eqx.Module]] = {}


def register_model(
    name: Optional[str] = None,
) -> Callable[[type[eqx.Module]], type[eqx.Module]]:
    """Decorator to register a model class under a serialisable string key.

    Registered names are used by :func:`load_model` and :func:`get_model_cls`
    to reconstruct the model architecture from a saved config. Collision
    checking prevents silent overwrites of core models.

    Example::

        @register_model("mynet")
        class MyNet(eqx.Module):
            ...

        model = load_model("mynet", path=Path("mynet.tar.lz4"))
    """

    def decorator(cls: type[eqx.Module]) -> type[eqx.Module]:
        if not issubclass(cls, eqx.Module):
            raise TypeError(
                f"Registered class must be a subclass of eqx.Module, got {type(cls)}"
            )
        registry_name = name.lower() if name else cls.__name__.lower()
        if registry_name in _MODEL_REGISTRY:
            raise ValueError(
                f"Cannot register '{registry_name}'. It is already registered "
                f"to {_MODEL_REGISTRY[registry_name]}."
            )
        _MODEL_REGISTRY[registry_name] = cls
        return cls

    return decorator


def get_model_cls(cls: str | type[eqx.Module]) -> type[eqx.Module]:
    """Resolve a model class from its registered string key or pass it through.

    Args:
        cls: A registered name (case-insensitive) or an ``eqx.Module`` subclass.

    Returns:
        The corresponding model class.

    Raises:
        ValueError: If ``cls`` is a string not present in the registry.
    """
    if not isinstance(cls, str):
        return cls

    cls_lower = cls.lower()

    # Experimental models are lazy-loaded to avoid importing heavy optional
    # dependencies (TensorFlow, SentencePiece) at module initialisation time.
    if cls_lower == "experimental.textencoder":
        from equimo.experimental.text import TextEncoder

        return TextEncoder

    if cls_lower not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model class: {cls!r}. "
            f"Available: {sorted(_MODEL_REGISTRY)}. "
            "Use register_model() to add custom models."
        )
    return _MODEL_REGISTRY[cls_lower]


_MODEL_REGISTRY["vit"] = em.VisionTransformer
_MODEL_REGISTRY["mlla"] = em.Mlla
_MODEL_REGISTRY["vssd"] = em.Vssd
_MODEL_REGISTRY["shvit"] = em.SHViT
_MODEL_REGISTRY["fastervit"] = em.FasterViT
_MODEL_REGISTRY["partialformer"] = em.PartialFormer
_MODEL_REGISTRY["iformer"] = em.IFormer
_MODEL_REGISTRY["mobilenetv3"] = em.MobileNetv3
_MODEL_REGISTRY["reduceformer"] = em.ReduceFormer


def _validate_identifier(identifier: str) -> None:
    """Raise ValueError if *identifier* contains characters unsafe for use in
    file paths or URLs (e.g. ``..``, ``/``, ``?``).
    """
    if not _SAFE_IDENTIFIER_RE.match(identifier):
        raise ValueError(
            f"Unsafe model identifier: {identifier!r}. "
            "Only alphanumeric characters, hyphens, and underscores are allowed."
        )


def save_model(
    path: Path,
    model: eqx.Module,
    model_config: dict,
    torch_hub_cfg: list[str] | dict | None = None,
    timm_cfg: list | None = None,
    compression: bool = True,
) -> None:
    """Save an Equinox model with its configuration and metadata to disk.

    Args:
        path: Target path. When *compression* is ``True`` and *path* does not
            end with ``.tar.lz4``, the suffix is appended automatically.
        model: The Equinox model to save. Saved dtype is preserved — bf16 models
            are serialised in bf16.
        model_config: Hyperparameter dictionary used to reconstruct the model.
        torch_hub_cfg: Optional torch-hub configuration (list or dict).
            Defaults to ``{}`` when ``None``.
        timm_cfg: Optional timm configuration list.
            Defaults to ``[]`` when ``None``.
        compression: If ``True`` (default), create a LZ4-compressed tar archive.
            If ``False``, write a plain directory.
    """
    # Guard against mutable-default aliasing from callers.
    torch_hub_cfg = torch_hub_cfg if torch_hub_cfg is not None else {}
    timm_cfg = timm_cfg if timm_cfg is not None else []

    logger.info(f"Saving model to {path}...")

    metadata = {
        "model_config": model_config,
        "torch_hub_cfg": torch_hub_cfg,
        "timm": timm_cfg,
        "jax_version": jax.__version__,
        "equinox_version": eqx.__version__,
        "equimo_version": __version__,
    }

    logger.debug(f"Metadata: {metadata}")

    if compression:
        logger.info("Compressing...")
        if not path.name.endswith(".tar.lz4"):
            path = path.with_name(path.name + ".tar.lz4")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with open(tmp_path / "metadata.json", "w") as f:
                json.dump(metadata, f)

            eqx.tree_serialise_leaves(tmp_path / "weights.eqx", model)

            path.parent.mkdir(parents=True, exist_ok=True)
            with lz4.frame.open(path, "wb") as f_out:
                with tarfile.open(fileobj=f_out, mode="w") as tar:
                    tar.add(tmp_path / "metadata.json", arcname="metadata.json")
                    tar.add(tmp_path / "weights.eqx", arcname="weights.eqx")
    else:
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f)
        eqx.tree_serialise_leaves(path / "weights.eqx", model)

    logger.info("Model successfully saved.")


def download(
    identifier: str,
    repository: str,
    timeout: int = 60,
) -> Path:
    """Download a model archive from a remote repository.

    Args:
        identifier: Unique model identifier. Must contain only alphanumeric
            characters, hyphens, and underscores (validated to prevent path
            traversal).
        repository: Base URL of the repository.
        timeout: HTTP request timeout in seconds. Defaults to 60.

    Returns:
        Local path to the downloaded (and cached) archive.

    Raises:
        ValueError: If *identifier* contains unsafe characters.
        requests.HTTPError: If the server returns a 4xx or 5xx response.
    """
    _validate_identifier(identifier)
    logger.info(f"Downloading {identifier}...")

    model = identifier.split("_")[0]
    url = f"{repository}/{model}/{identifier}.tar.lz4"
    path = Path(f"~/.cache/equimo/{model}/{identifier}.tar.lz4").expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        logger.info("Archive already downloaded, using cached file.")
        return path

    tmp_path = path.with_suffix(".tmp")
    fd, tmp_str = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_str)
    try:
        with requests.get(url, stream=True, timeout=timeout, verify=True) as res:
            res.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in res.iter_content(chunk_size=65_536):
                    f.write(chunk)
        tmp_path.rename(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    return path


def load_model(
    cls: str | type[eqx.Module],
    identifier: str | None = None,
    path: Path | None = None,
    repository: str = DEFAULT_REPOSITORY_URL,
    inference_mode: bool = True,
    **model_kwargs,
) -> eqx.Module:
    """Load an Equinox model from a local path or remote repository.

    Args:
        cls: Model class or registered name (case-insensitive). Use
            :func:`register_model` to add custom models.
        identifier: Remote model identifier for downloading. Mutually
            exclusive with *path*.
        path: Local path to load the model from. Mutually exclusive
            with *identifier*.
        repository: Base URL for model download.
            Defaults to :data:`DEFAULT_REPOSITORY_URL`.
        inference_mode: Pass ``True`` (default) to disable dropout for
            evaluation; ``False`` to keep training behaviour.
        **model_kwargs: Extra keyword arguments merged into the saved
            ``model_config``, overriding stored values.

    Returns:
        Loaded model with weights deserialised. Dtype is whatever was
        stored (bf16 models are loaded as bf16).

    Raises:
        ValueError: If both or neither of *identifier*/*path* are given,
            or if *cls* is not a registered model name.
    """
    if identifier is None and path is None:
        raise ValueError(
            "Both `identifier` and `path` are None. Please provide one of them."
        )
    if identifier is not None and path is not None:
        raise ValueError(
            "Both `identifier` and `path` are defined. Please provide only one of them."
        )

    if identifier is not None:
        path = download(identifier, repository)

    load_path = path
    model_cls = get_model_cls(cls)
    logger.info(f"Loading a {model_cls.__name__} model...")

    if path.suffixes == [".tar", ".lz4"]:
        logger.info("Decompressing...")
        decompressed_dir = path.with_suffix("").with_suffix("")
        sentinel = decompressed_dir / ".complete"

        # Re-extract if sentinel is missing or stale, survives interrupted extractions
        if not sentinel.exists() or (sentinel.stat().st_mtime < path.stat().st_mtime):
            tmp_dir = Path(
                tempfile.mkdtemp(dir=decompressed_dir.parent, prefix=".tmp_extract_")
            )
            try:
                with lz4.frame.open(path, "rb") as f_in:
                    with tarfile.open(fileobj=f_in, mode="r") as tar:
                        tar.extractall(tmp_dir, filter="data")
                # Write sentinel INSIDE the temp dir before the atomic swap
                (tmp_dir / ".complete").touch()
                if decompressed_dir.exists():
                    shutil.rmtree(decompressed_dir)
                tmp_dir.rename(decompressed_dir)
            except BaseException:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise

        load_path = decompressed_dir

    with open(load_path / "metadata.json", "r") as f:
        metadata = json.load(f)

    logger.debug(f"Metadata: {metadata}")

    model_eqm_version = metadata.get("equimo_version", "0.2.0")
    if Version.parse(model_eqm_version) > Version.parse(__version__):
        logger.warning(
            f"The model you are importing was packaged with equimo "
            f"v{model_eqm_version}, but you have equimo v{__version__}. "
            "You may face unexpected errors. Please consider updating equimo."
        )

    kwargs = metadata["model_config"] | model_kwargs
    model = model_cls(**kwargs, key=jax.random.PRNGKey(42))

    model = eqx.tree_deserialise_leaves(load_path / "weights.eqx", model)
    model = eqx.nn.inference_mode(model, inference_mode)

    logger.info("Model loaded successfully.")
    return model


def _center_crop_square(array: Float[Array, "..."]) -> Float[Array, "..."]:
    """Center-crop a H×W(×C) array to a square of side min(H, W).

    Args:
        array: Array with shape ``(H, W)`` or ``(H, W, C)``.

    Returns:
        Center-cropped array with shape ``(M, M)`` or ``(M, M, C)``
        where ``M = min(H, W)``.
    """
    if array.ndim < 2:
        raise ValueError("Input array must have at least 2 dimensions (H, W[, C]).")
    h, w = array.shape[:2]
    if h == w:
        return array
    m = min(h, w)
    top = (h - m) // 2
    left = (w - m) // 2
    return array[top : top + m, left : left + m, ...]


def load_image(
    path: str,
    mean: Optional[list[float]] = None,
    std: Optional[list[float]] = None,
    size: Optional[int] = None,
    center_crop: bool = False,
) -> Float[Array, "channels height width"]:
    """Load an image from disk and optionally preprocess it.

    The image is always converted to RGB (3-channel) before processing.
    The returned array is in channel-first format (C, H, W) and dtype
    ``float32``, normalised to [0, 1] before any mean/std shift.

    Args:
        path: Path to the image file.
        mean: Per-channel mean for normalisation, e.g. ``[0.485, 0.456, 0.406]``.
        std: Per-channel standard deviation, e.g. ``[0.229, 0.224, 0.225]``.
        size: Resize both spatial dimensions to this value (square).
        center_crop: If ``True``, center-crop to a square before resizing.

    Returns:
        Float32 array of shape ``(3, H, W)``.

    Raises:
        ImportError: If Pillow is not installed.
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL is needed to be able to load images.")

    with open(path, "rb") as fd:
        image_bytes = io.BytesIO(fd.read())
        pil_image = Image.open(image_bytes).convert("RGB")

        array = jnp.array(pil_image).astype(jnp.float32) / 255.0

        if center_crop:
            array = _center_crop_square(array)

        if size is not None:
            array = jax.image.resize(
                array, (size, size, array.shape[2]), method="bilinear"
            )

        if mean is not None and std is not None:
            mean_arr = jnp.array(mean)[None, None, :]
            std_arr = jnp.array(std)[None, None, :]
            array = (array - mean_arr) / std_arr

    return array.transpose(2, 0, 1)
