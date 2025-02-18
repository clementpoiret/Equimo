import json
import tarfile
import tempfile
from pathlib import Path

import equinox as eqx
import jax
import lz4.frame
import requests
from loguru import logger

import equimo.models as em

DEFAULT_REPOSITORY_URL = (
    "https://huggingface.co/poiretclement/equimo/resolve/main/models/default"
)


def save_model(
    path: Path,
    model: eqx.Module,
    model_config: dict,
    torch_hub_cfg: list[str],
    compression: bool = True,
):
    """Save an Equinox model with its configuration and metadata to disk.

    Args:
        path (Path): Target path where the model will be saved. If compression is True
            and path doesn't end with '.tar.lz4', it will be automatically appended.
        model (eqx.Module): The Equinox model to be saved.
        model_config (dict): Configuration dictionary containing model hyperparameters.
        torch_hub_cfg (list[str]): List of torch hub configuration parameters.
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
        "jax_version": jax.__version__,
        "equinox_version": eqx.__version__,
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

    url = f"{repository}/{identifier}.tar.lz4"
    path = Path(f"~/.cache/equimo/{identifier}.tar.lz4").expanduser()

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
        case _:
            raise ValueError(f"Unknown model class: {cls}")

    # Reconstruct model skeleton
    model = model_cls(**metadata["model_config"], key=jax.random.PRNGKey(42))

    # Load weights
    model = eqx.tree_deserialise_leaves(load_path / "weights.eqx", model)

    logger.info("Model loaded successfully.")

    return model
