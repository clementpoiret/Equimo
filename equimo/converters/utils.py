import re
from typing import Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.tree_util import GetAttrKey, SequenceKey
from loguru import logger


def stringify_name(path: Tuple) -> str:
    stringified = []
    for p in path:
        if isinstance(p, GetAttrKey):
            stringified.append(p.name)
        if isinstance(p, SequenceKey):
            stringified.append(str(p.idx))
    return ".".join(stringified)


def expand_torch_tensor(tensor, pos: str, n: int):
    padding = [None] * n
    match pos:
        case "before":
            return tensor[*padding, ...]
        case "after":
            return tensor[..., *padding]
        case _:
            raise ValueError(
                f"Invalid `pos`, expected one of [`before`, `after`], got: {pos}"
            )


def convert_params_from_torch_hub(
    jax_model: eqx.Module,
    torch_hub_cfg: list[str],
    replace_cfg: Dict[str, str],
    expand_cfg: Dict[str, list],
    squeeze_cfg: Dict[str, int | None],
    whitelist: list[str],
    strict: bool = True,
):
    """
    Load weights from a torch hub model into an Equinox module.

    Args:
        jax_model (eqx.Module): A preexisting Jax model corresponding to the checkpoint to download.
        torch_hub_cfg (Tuple[str]): Arguments passed to `torch.hub.load()`.
        transpose_whitelist (Set[str]): Parameters to exclude from format conversion.
        strict (bool): Whether to crash on missing parameters one of the models.
    """
    try:
        import torch
    except:
        raise ImportError("`torch` not available")

    # Load the pytorch model from torch hub
    torch_model = torch.hub.load(*torch_hub_cfg)
    torch_params = dict(torch_model.named_parameters())

    # Extract the parameters from the defined Jax model
    jax_params = eqx.filter(jax_model, eqx.is_array)
    # _, jax_params, _ = nnx.split(jax_model, nnx.Param, ...)
    jax_params_flat, jax_param_pytree = jax.tree_util.tree_flatten_with_path(jax_params)

    torch_params_flat = []
    for path, param in jax_params_flat:
        # Match the parameters' path of pytorch
        param_path = stringify_name(path)
        param_path = re.sub(r"\.scale|.kernel", ".weight", param_path)

        for old, new in replace_cfg.items():
            param_path = param_path.replace(old, new)

        shape = param.shape

        if param_path not in torch_params:
            _msg = f"{param_path} ({shape}) not found in PyTorch model."
            if strict:
                logger.error(_msg)
                raise AttributeError(_msg)

            logger.warning(f"{_msg} Appending `None` to flat param list.")
            torch_params_flat.append(None)
            continue

        logger.info(f"Converting {param_path}...")
        torch_param = torch_params[param_path]

        if param_path in expand_cfg:
            torch_param = expand_torch_tensor(torch_param, *expand_cfg[param_path])
        if param_path in squeeze_cfg:
            torch_param = torch.squeeze(torch_param, dim=squeeze_cfg[param_path])

        if shape != torch_param.shape:
            _msg = f"`{param_path}`: expected shape ({shape}) does not match its pytorch implementation ({torch_param.shape})."
            logger.error(_msg)
            raise ValueError(_msg)

        torch_params_flat.append(jnp.asarray(torch_param.detach().numpy()))
        _ = torch_params.pop(param_path)

    loaded_params = jax.tree_util.tree_unflatten(jax_param_pytree, torch_params_flat)

    for path, param in torch_params.items():
        logger.warning(
            f"PyTorch parameters `{path}` ({param.shape}) were not converted."
        )
        if strict and path not in whitelist:
            _msg = f"The PyTorch model contains parameters ({path}) that do not have a Jax counterpart."
            logger.error(_msg)
            raise AttributeError(_msg)

    return loaded_params


def convert_torch_to_equinox(
    jax_model: eqx.Module,
    torch_hub_cfg: list[str],
    replace_cfg: dict = {},
    expand_cfg: dict = {},
    squeeze_cfg: dict = {},
    whitelist: list[str] = [],
    strict: bool = True,
) -> eqx.Module:
    """
    Convert a PyTorch model from torch.hub to Equinox format.

    Args:
        jax_model: The Equinox model
        torch_hub_cfg: [repo, model_name] for torch.hub.load
        replace_cfg: Dict of parameter name replacements
        expand_cfg: Dict of dimensions to expand
        squeeze_cfg: Dict of dimensions to squeeze
        whitelist: List of parameters to keep from JAX model
        strict: Wether to raise an issue if not all weights are converted

    Returns:
        eqx.Module: Converted Equinox model in inference mode
    """
    dynamic, static = eqx.partition(jax_model, eqx.is_array)
    converted_params = convert_params_from_torch_hub(
        dynamic,
        torch_hub_cfg,
        replace_cfg,
        expand_cfg,
        squeeze_cfg,
        whitelist,
        strict,
    )

    return eqx.nn.inference_mode(eqx.combine(converted_params, static), value=True)
