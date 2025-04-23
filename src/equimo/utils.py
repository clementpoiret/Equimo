import typing as t
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Float

_ArrayLike = t.Union[np.ndarray, jnp.ndarray]


def normalize(x, order: int = 2):
    return x / np.linalg.norm(x, ord=order, axis=-1, keepdims=True).clip(min=1e-3)


class PCAVisualizer:
    """PCA visualizer.

    Taken as-is from https://github.com/google-deepmind/tips/blob/72820c1841f973c9543d9c95c5ff2262ec621955/scenic/utils/feature_viz.py#L32
    """

    def __init__(
        self, features: _ArrayLike, n_samples: int = 100000, n_components: int = 3
    ) -> None:
        """Creates a PCA object for visualizing features of shape [..., F]."""
        try:
            from sklearn import decomposition
        except ImportError:
            raise ImportError(
                "You need sklearn to use the PCAVisualizer, install it using `uv add equimo[viz]`"
            )

        features = np.array(features)
        pca_object = decomposition.PCA(n_components=n_components)
        features = features.reshape([-1, features.shape[-1]])
        features = features[np.random.randint(0, features.shape[0], n_samples), :]
        pca_object.fit(features)
        self.pca_object = pca_object
        self.n_components = n_components

    def __call__(self, features: _ArrayLike) -> np.ndarray:
        """Apply PCA to features of shape [..., F]."""
        features = np.array(features)
        features_pca = self.pca_object.transform(
            features.reshape([-1, features.shape[-1]])
        ).reshape(features.shape[:-1] + (self.n_components,))
        return normalize(features_pca) * 0.5 + 0.5


def plot_image_and_feature_map(
    image,
    feature_map,
    save_path,
    image_title="Image",
    map_title="Feature Map",
    fontsize=16,
):
    """
    Plots an image and its feature map side by side and saves the figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Matplotlib is required to plot an image")

    feature_map_normalized = (feature_map - feature_map.min()) / (
        feature_map.max() - feature_map.min()
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(image)
    ax1.set_title(image_title, fontsize=fontsize)
    ax1.axis("off")

    ax2.imshow(feature_map_normalized, cmap="gray")
    ax2.set_title(map_title, fontsize=fontsize)
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def make_2tuple(x):
    """Convert input into a 2-tuple.

    Args:
        x: Input value, either an integer or a 2-tuple

    Returns:
        tuple: If input is integer, returns (x,x). If input is 2-tuple, returns it unchanged.

    Raises:
        AssertionError: If input is tuple but not length 2, or if input is not int or tuple
    """
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


def to_list(obj, n):
    """Convert an object to a list of length n by repeating it or validating existing list.

    Args:
        obj: Input object or list
        n: Desired length of output list

    Returns:
        List of length n containing obj repeated n times if obj is not a list,
        or the original list if it's already length n

    Raises:
        AssertionError: If obj is a list but its length doesn't match n
    """
    if isinstance(obj, list):
        if len(obj) == n:
            return obj
        else:
            raise AssertionError(
                f"obj (list of len {len(obj)}) should have a size of {n}"
            )

    return [obj] * n


def nearest_power_of_2_divisor(dim: int, max: int):
    """Find the largest power of 2 that divides dim, up to a maximum value.

    Args:
        dim: The number to find divisors for
        max: Maximum value to consider (must be a power of 2)

    Returns:
        int: Largest power of 2 that divides dim, not exceeding max

    Example:
        >>> nearest_power_of_2_divisor(24, 32)
        8  # because 8 is the largest power of 2 <= 32 that divides 24
    """
    power = 1
    nearest = 1
    while power <= max:
        if dim % power == 0:
            nearest = power
        power *= 2
    return nearest


@partial(
    jax.jit,
    static_argnames=[
        "pool_type",
        "num_prefix_tokens",
        "reduce_include_prefix",
    ],
)
def pool_sd(
    x: Float[Array, "seqlen dim"],
    pool_type: str = "token",
    num_prefix_tokens: int = 1,
    reduce_include_prefix: bool = False,
):
    """Pool sequence dimension using various strategies.

    Args:
        x: Input tensor of shape (sequence_length, dimension)
        pool_type: Pooling strategy to use:
            - "token": Use first token (typically CLS token)
            - "avg": Average pooling
            - "max": Max pooling
            - "avgmax": Average of max and mean pooling
            - "": No pooling (return input unchanged)
        num_prefix_tokens: Number of special tokens at start of sequence
        reduce_include_prefix: Whether to include prefix tokens in pooling

    Returns:
        Pooled tensor. If pool_type is "token", returns vector of size dim.
        For other pool types, returns reduced tensor according to the strategy.

    Raises:
        ValueError: If pool_type is not one of the supported values
    """
    if not pool_type:
        return x

    if pool_type == "token":
        x = x[0]  # class token
    else:
        x = x if reduce_include_prefix else x[num_prefix_tokens:]
        match pool_type:
            case "avg":
                x = jnp.mean(x, axis=0)
            case "avgmax":
                x = 0.5 * (jnp.max(x, axis=0) + jnp.mean(x, axis=0))
            case "max":
                x = jnp.max(x, axis=0)
            case _:
                raise ValueError(f"Unknown pool type {pool_type}")

    return x


def count_params(model: eqx.Module):
    num_params = sum(
        x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    )
    return num_params / 1_000_000


def cost_analysis(model: eqx.Module, input_example: Float[Array, "..."]):
    """Estimates the memory usage, flops, and #params of a model's forward pass.

    This function JIT-compiles the model's forward pass for a given input,
    retrieves the cost analysis, and extracts the estimated bytes accessed,
    converting it to Mebibytes (MiB), the flops converting it to GigaFLOPs
    (GFLOPs), and the number of parameters in millions.

    Args:
        model: The Equinox model.
        x: An example input tensor for the model.

    Returns:
        A dict containing the relevant information.
    """
    key = jr.PRNGKey(42)

    @jax.jit
    def fpass(x):
        return model(x, inference=True, key=key)

    analysis: dict | list[dict] = fpass.lower(input_example).compile().cost_analysis()
    cost_dict: dict = analysis[0] if isinstance(analysis, list) else analysis

    # Memory
    memory_mib = cost_dict.get("bytes accessed", 0.0) / (1024 * 1024)

    # Flops
    gflops = cost_dict.get("flops", 0.0) / 1_000_000_000

    # Params
    mparams = count_params(model)

    return {
        "memory_mib": memory_mib,
        "gflops": gflops,
        "mparams": mparams,
    }
