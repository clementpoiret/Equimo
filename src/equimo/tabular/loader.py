"""Load pretrained TabPFN v3 (Prior Labs) checkpoints into the Equinox model.

The Equinox module tree mirrors the PyTorch attribute names, so loading is a
direct path -> state-dict-key match (no transposes: eqx ``Linear`` weight is
``(out, in)`` like torch). torch is only needed to read the ``.ckpt``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from jaxtyping import PRNGKeyArray

from equimo.conversion.utils import stringify_name
from equimo.tabular.model import TabPFNV3

# torch buffers / params with no Equinox counterpart (recomputed or unused).
_IGNORED_TORCH = ("regression_borders", "column_aggregator.rope.freqs")

# config keys consumed by TabPFNV3.__init__ (others in the ckpt config are ignored).
_CONFIG_KEYS = (
    "max_num_classes",
    "embed_dim",
    "dist_embed_num_blocks",
    "dist_embed_num_heads",
    "dist_embed_num_inducing_points",
    "feature_group_size",
    "feat_agg_num_blocks",
    "feat_agg_num_heads",
    "feat_agg_num_cls_tokens",
    "feat_agg_rope_base",
    "use_rope",
    "nlayers",
    "icl_num_heads",
    "icl_num_kv_heads_test",
    "decoder_head_dim",
    "decoder_num_heads",
    "decoder_use_softmax_scaling",
    "ff_factor",
    "softmax_scaling_mlp_hidden_dim",
    "use_nan_indicators",
)


def build_model(config: dict, *, key: PRNGKeyArray) -> TabPFNV3:
    """Instantiate a (randomly initialised) TabPFNV3 from a checkpoint config dict."""
    kwargs = {k: config[k] for k in _CONFIG_KEYS if k in config}
    return TabPFNV3(**kwargs, key=key)


def load_state_dict(
    model: TabPFNV3,
    state_dict: dict,
    *,
    strict: bool = True,
) -> TabPFNV3:
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


def from_pretrained(ckpt_path: str, *, key: PRNGKeyArray | None = None) -> TabPFNV3:
    """Build a TabPFNV3 from a ``.ckpt`` file and load its weights."""
    import torch  # local import: only needed to read the checkpoint

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if key is None:
        key = jax.random.PRNGKey(0)
    model = build_model(ckpt["config"], key=key)
    return load_state_dict(model, ckpt["state_dict"])
