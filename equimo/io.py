import json
import tarfile
import tempfile
from pathlib import Path
from loguru import logger

import equinox as eqx
import jax
import lz4.frame
import equimo.models as em


def save_model(
    path: Path,
    model: eqx.Module,
    model_config: dict,
    torch_hub_cfg: list[str],
    compression: bool = True,
):
    """Save model with hyperparameters using Equinox serialization."""

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


def load_model(
    path: Path, cls: str, key: jax.Array = jax.random.PRNGKey(42)
) -> eqx.Module:
    """Load model from serialized format, handling both directories and archives."""
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
        case _:
            raise ValueError(f"Unknown model class: {cls}")

    # Reconstruct model skeleton
    model = model_cls(**metadata["model_config"], key=key)

    # Load weights
    model = eqx.tree_deserialise_leaves(load_path / "weights.eqx", model)

    logger.info("Model loaded successfully.")

    return model
