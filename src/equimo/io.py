import json
import io
import tarfile
import tempfile
from pathlib import Path
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import lz4.frame
import requests
from loguru import logger
from semver import Version

import equimo.models as em
from equimo import __version__

DEFAULT_REPOSITORY_URL = (
    "https://huggingface.co/poiretclement/equimo/resolve/main/models/default"
)


def save_model(
    path: Path,
    model: eqx.Module,
    model_config: dict,
    torch_hub_cfg: list[str] = [],
    timm_cfg: list = [],
    compression: bool = True,
):
    """Save an Equinox model with its configuration and metadata to disk.

    Args:
        path (Path): Target path where the model will be saved. If compression is True
            and path doesn't end with '.tar.lz4', it will be automatically appended.
        model (eqx.Module): The Equinox model to be saved.
        model_config (dict): Configuration dictionary containing model hyperparameters.
        torch_hub_cfg (list[str]): List of torch hub configuration parameters.
        timm_cfg (list[str]): List of timm configuration parameters.
        compression (bool, optional): Whether to compress the saved model using LZ4.
            Defaults to True.

    The function saves:
        - Model weights using Equinox serialization
        - Metadata including model configuration, torch hub config, and version info
        - If compression=True: Creates a .tar.lz4 archive containing both files
        - If compression=False: Creates a directory containing both files
    """

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

            # Save model weights
            eqx.tree_serialise_leaves(tmp_path / "weights.eqx", model)

            # Create compressed archive
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

    logger.info("Model succesfully saved.")


@logger.catch
def download(identifier: str, repository: str) -> Path:
    """Download a model archive from a specified repository.

    Args:
        identifier (str): Unique identifier for the model to download.
        repository (str): Base URL of the repository containing the model.

    Returns:
        Path: Local path to the downloaded model archive.

    The function:
        - Constructs the download URL using the repository and identifier
        - Checks for existing cached file in ~/.cache/equimo/
        - Downloads and saves the model if not cached
        - Verifies the download using HTTPS
        - Raises HTTP errors if download fails
    """

    logger.info(f"Downloading {identifier}...")

    model = identifier.split("_")[0]
    url = f"{repository}/{model}/{identifier}.tar.lz4"
    path = Path(f"~/.cache/equimo/{model}/{identifier}.tar.lz4").expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        logger.info("Archive already downloaded, using cached file.")
        return path

    res = requests.get(url, verify=True)
    res.raise_for_status()

    with open(path.absolute(), "wb") as f:
        f.write(res.content)

    return path


@logger.catch
def load_model(
    cls: str,
    identifier: str | None = None,
    path: Path | None = None,
    repository: str = DEFAULT_REPOSITORY_URL,
    inference_mode: bool = True,
    **model_kwargs,
) -> eqx.Module:
    """Load an Equinox model from either a local path or remote repository.

    Args:
        cls (str): Model class identifier. Must be one of: 'vit', 'mlla', 'vssd',
            'shvit', 'fastervit', 'partialformer'.
        identifier (str | None, optional): Remote model identifier for downloading.
            Mutually exclusive with path. Defaults to None.
        path (Path | None, optional): Local path to load model from.
            Mutually exclusive with identifier. Defaults to None.
        repository (str, optional): Base URL for model download.
            Defaults to DEFAULT_REPOSITORY_URL.
        inference_mode (bool): Disables dropouts if True.
            Defaults to True.
        model_kwargs: kwargs passed to model instanciation. Overrides metadatas.

    Returns:
        eqx.Module: Loaded and initialized model with weights.

    Raises:
        ValueError: If both identifier and path are None or if both are provided.
        ValueError: If cls is not one of the supported model types.

    The function:
        - Downloads model if identifier is provided
        - Handles both compressed (.tar.lz4) and uncompressed formats
        - Loads model configuration and metadata
        - Reconstructs model architecture and loads weights
        - Supports caching of downloaded and decompressed files
    """

    if identifier is None and path is None:
        raise ValueError(
            "Both `identifier` and `path` are None. Please provide one of them."
        )
    if identifier and path:
        raise ValueError(
            "Both `identifier` and `path` are defined. Please provide only one of them."
        )

    if identifier:
        path = download(identifier, repository)

    load_path = path

    logger.info(f"Loading a {cls} model...")

    if path.suffixes == [".tar", ".lz4"]:
        logger.info("Decompressing...")
        # Handle compressed archive
        decompressed_dir = path.with_suffix("").with_suffix("")  # Remove .tar.lz4

        # Check if we need to decompress
        if not decompressed_dir.exists() or (
            decompressed_dir.stat().st_mtime < path.stat().st_mtime
        ):
            decompressed_dir.mkdir(parents=True, exist_ok=True)
            with lz4.frame.open(path, "rb") as f_in:
                with tarfile.open(fileobj=f_in, mode="r") as tar:
                    tar.extractall(decompressed_dir)

        load_path = decompressed_dir

    # Load metadata and model
    with open(load_path / "metadata.json", "r") as f:
        metadata = json.load(f)

    logger.debug(f"Metadata: {metadata}")

    model_eqm_version = metadata.get("equimo_version", "0.2.0")
    if Version.parse(model_eqm_version) > Version.parse(__version__):
        logger.warning(
            f"The model you are importing was packaged with equimo v{model_eqm_version}, but you have equimo v{__version__}. You may face unexpected errors. Please consider updating equimo."
        )

    # Class resolution
    match cls:
        case "vit":
            model_cls = em.VisionTransformer
        case "mlla":
            model_cls = em.Mlla
        case "vssd":
            model_cls = em.Vssd
        case "shvit":
            model_cls = em.SHViT
        case "fastervit":
            model_cls = em.FasterViT
        case "partialformer":
            model_cls = em.PartialFormer
        case "experimental.textencoder":
            from equimo.experimental.text import TextEncoder

            model_cls = TextEncoder
        case _:
            raise ValueError(f"Unknown model class: {cls}")

    # Reconstruct model skeleton
    kwargs = metadata["model_config"] | model_kwargs
    model = model_cls(**kwargs, key=jax.random.PRNGKey(42))

    # Load weights and set inference mode
    model = eqx.tree_deserialise_leaves(load_path / "weights.eqx", model)
    model = eqx.nn.inference_mode(model, inference_mode)

    logger.info("Model loaded successfully.")

    return model


def load_image(
    path: str,
    mean: Optional[list[float]] = None,
    std: Optional[list[float]] = None,
    size: Optional[int] = None,
):
    """Load an image and perform minor preprocessing.

    Args:
        path (str): Path of the image.
        mean (list, optional): Channel mean for normalization.
        std (list, optional): Channel std for normalization.
        size (int, optional): Size to which resize the image.

    Returns:
        jnp.array: The loaded image.

    Raises:
        ImportError: If PIL can't be loaded.
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL is needed to be able to load images.")

    with open(path, "rb") as fd:
        image_bytes = io.BytesIO(fd.read())
        pil_image = Image.open(image_bytes)

        array = jnp.array(pil_image).astype(jnp.float32) / 255.0
        if size is not None:
            array = jax.image.resize(array, (size, size, 3), method="bilinear")

        if mean is not None and std is not None:
            mean = jnp.array(mean)[None, None, :]
            std = jnp.array(std)[None, None, :]

            array = (array - mean) / std

    return array.transpose(2, 0, 1)
