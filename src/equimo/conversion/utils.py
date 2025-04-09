import re
from typing import Any, Dict, Literal, Optional, Tuple

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


def convert_params_from_torch(
    jax_model: eqx.Module,
    replace_cfg: Dict[str, str],
    expand_cfg: Dict[str, list],
    squeeze_cfg: Dict[str, int | None],
    whitelist: list[str],
    strict: bool = True,
    source: Literal["torchhub", "timm", "custom"] = "torchhub",
    torch_hub_cfg: Optional[list[str]] = None,
    torch_model=None,
    timm_cfg: Optional[list] = None,
    return_torch: bool = False,
):
    """
    Load weights from a torch hub model into an Equinox module.

    Args:
        jax_model (eqx.Module): A preexisting Jax model corresponding to the checkpoint to download.
        torch_hub_cfg (Tuple[str]): Arguments passed to `torch.hub.load()`.
        replace_cfg (Dict[str, str]): Rename parameters from key to value.
        expand_cfg (Dict[str, list]): Config to reshape params, see `expand_torch_tensor`
        sqeeze_cfg (Dict[str, int|None]): Config to squeeze tensors, opposite of expand.
        whitelist (Set[str]): Parameters to exclude from format conversion.
        strict (bool): Whether to crash on missing parameters one of the models.
        source (str): Torch Hub or timm.
        torch_hub_cfg (Optional[list]): args to pass to `torch.hub.load`.
        torch_model [torch.nn.Module]: Custom torch model
        timm_cfg (Optional[list]): args to pass to `timm.create_model`.
        return_torch (bool): Return both jax and torch models.
    """
    try:
        import timm
        import torch
    except:
        raise ImportError("`torch` not available")

    # Load the pytorch model
    match source:
        case "custom":
            if torch_model is None:
                raise ValueError(
                    "The `custom` source is selected but `torch_model` is None."
                )
        case "torchhub":
            if torch_hub_cfg is None:
                raise ValueError(
                    "The `torchhub` source is selected but `torch_hub_cfg` is None."
                )
            torch_model = torch.hub.load(*torch_hub_cfg)
        case "timm":
            if timm_cfg is None:
                raise ValueError(
                    "The `timm` source is selected but `timm_cfg` is None."
                )
            torch_model = timm.create_model(*timm_cfg)

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

    if return_torch:
        return loaded_params, torch_model
    return loaded_params


def convert_torch_to_equinox(
    jax_model: eqx.Module,
    replace_cfg: dict = {},
    expand_cfg: dict = {},
    squeeze_cfg: dict = {},
    whitelist: list[str] = [],
    strict: bool = True,
    source: Literal["torchhub", "timm"] = "torchhub",
    torch_hub_cfg: Optional[list[str]] = None,
    torch_model=None,
    timm_cfg: Optional[list] = None,
    return_torch: bool = False,
) -> eqx.Module | Tuple[eqx.Module, Any]:
    """
    Convert a PyTorch model from torch.hub to Equinox format.

    Args:
        jax_model: The Equinox model
        replace_cfg: Dict of parameter name replacements
        expand_cfg: Dict of dimensions to expand
        squeeze_cfg: Dict of dimensions to squeeze
        whitelist: List of parameters to keep from JAX model
        strict: Wether to raise an issue if not all weights are converted
        source (str): Torch Hub or timm.
        torch_hub_cfg: [repo, model_name] for torch.hub.load
        torch_model [torch.nn.Module]: Custom torch model
        timm_cfg (Optional[list]): args to pass to `timm.create_model`.
        return_torch (bool): Return both jax and torch models.

    Returns:
        eqx.Module: Converted Equinox model in inference mode
    """
    dynamic, static = eqx.partition(jax_model, eqx.is_array)
    if return_torch:
        converted_params, torch_model = convert_params_from_torch(
            dynamic,
            replace_cfg,
            expand_cfg,
            squeeze_cfg,
            whitelist,
            strict,
            source,
            torch_hub_cfg,
            torch_model,
            timm_cfg,
            return_torch,
        )

        return eqx.nn.inference_mode(
            eqx.combine(converted_params, static), value=True
        ), torch_model.eval()
    else:
        converted_params = convert_params_from_torch(
            dynamic,
            replace_cfg,
            expand_cfg,
            squeeze_cfg,
            whitelist,
            strict,
            source,
            torch_hub_cfg,
            torch_model,
            timm_cfg,
            return_torch,
        )

        return eqx.nn.inference_mode(eqx.combine(converted_params, static), value=True)
