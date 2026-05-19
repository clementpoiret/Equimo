import io
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def _center_crop_square(array: Float[Array, "..."]) -> Float[Array, "..."]:
    """Center-crop a HxW(xC) array to a square of side min(H, W)."""
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
    """Load an RGB image as a channel-first float32 JAX array."""
    try:
        from PIL import Image  # ty: ignore[unresolved-import]
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
