"""State-dict helpers for TabPFN.

This module is intentionally minimal after the tabular API refactor. It expects
state-dict keys to match the current Equinox module tree.
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from jaxtyping import PRNGKeyArray

from equimo.conversion.utils import stringify_name
from equimo.tabular.models.tabpfn import TabPFN

# Torch buffers / params with no Equinox counterpart (recomputed or unused).
_IGNORED_TORCH = ("regression_borders", "column_aggregator.rope.freqs")

# Config keys consumed by TabPFN.__init__.
_CONFIG_KEYS = (
    "num_classes",
    "dim",
    "depths",
    "num_heads",
    "mlp_ratio",
    "feature_group_size",
    "num_inducing_points",
    "num_cls_tokens",
    "num_kv_heads_test",
    "decoder_head_dim",
    "decoder_num_heads",
    "decoder_use_softmax_scaling",
    "use_rope",
    "rope_base",
    "scaling_mlp_hidden_dim",
    "use_nan_indicators",
    "drop_path_rate",
    "drop_path_uniform",
    "context_block",
    "preprocessor_layer",
    "label_embedding_layer",
    "decoder_layer",
    "act_layer",
    "norm_layer",
    "eps",
)


def build_model(config: dict, *, key: PRNGKeyArray) -> TabPFN:
    """Instantiate a randomly initialised TabPFN from a config dict."""
    kwargs = {k: config[k] for k in _CONFIG_KEYS if k in config}
    return TabPFN(**kwargs, key=key)


def load_state_dict(
    model: TabPFN,
    state_dict: dict,
    *,
    strict: bool = True,
) -> TabPFN:
    """Copy a torch ``state_dict`` into ``model`` by matching pytree paths to keys.

    Returns the model in inference mode. ``state_dict`` values may be torch
    tensors or numpy arrays.
    """

    def to_np(t):
        return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)

    dynamic, static = eqx.partition(model, eqx.is_array)
    flat, treedef = jax.tree_util.tree_flatten_with_path(dynamic)

    new_leaves, used = [], set()
    for path, leaf in flat:
        name = stringify_name(path)
        if name not in state_dict:
            raise KeyError(f"{name} {tuple(leaf.shape)} missing from state_dict")
        arr = jnp.asarray(to_np(state_dict[name]))
        if arr.shape != leaf.shape:
            raise ValueError(
                f"{name}: jax shape {tuple(leaf.shape)} != torch {tuple(arr.shape)}"
            )
        new_leaves.append(arr)
        used.add(name)

    leftover = [
        k for k in state_dict if k not in used and not k.startswith(_IGNORED_TORCH)
    ]
    if leftover and strict:
        raise KeyError(f"Unconverted torch params: {leftover}")

    loaded = jax.tree_util.tree_unflatten(treedef, new_leaves)
    model = eqx.combine(loaded, static)
    return eqx.nn.inference_mode(model, value=True)


def from_pretrained(ckpt_path: str, *, key: PRNGKeyArray | None = None) -> TabPFN:
    """Build a TabPFN from a ``.ckpt`` file and load its weights."""
    import torch  # local import: only needed to read the checkpoint

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if key is None:
        key = jax.random.PRNGKey(0)
    model = build_model(ckpt["config"], key=key)
    return load_state_dict(model, ckpt["state_dict"])
