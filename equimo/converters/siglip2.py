from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import equimo.models as em
from equimo.converters.utils import convert_torch_to_equinox
from equimo.io import load_model, save_model


def compare(j, t) -> float:
    j = np.array(j)
    t = t.squeeze().detach().numpy()
    return float(np.mean(np.abs(j - t)))


def main():
    try:
        import torch
    except:
        raise ImportError("`torch` not available")

    key = jax.random.PRNGKey(42)
    siglip2_config = {
        "img_size": 384,
        "in_channels": 3,
        "dim": 1536,
        "patch_size": 16,
        "num_heads": [16],
        "depths": [40],
        "mlp_ratio": 4,
        "num_classes": 0,
        "use_mask_token": False,
        "reg_tokens": 0,
        "class_token": False,
        "no_embed_class": True,
        "init_values": None,
        "eps": 1e-6,
        "dynamic_img_size": False,
        # "act_layer": "exactgelu",
        "act_layer": "gelu",
    }

    siglip2 = em.VisionTransformer(
        **siglip2_config,
        key=key,
    )

    timm_cfg = ["vit_giantopt_patch16_siglip_gap_384.v2_webli", True]
    _n = timm_cfg[0].split("_")
    mname = f"siglip2_vitgiantopt{_n[2][-2:]}_{_n[-2][:3]}"
    # mname = f"siglip2_vit{_n[1][0]}{_n[2][-2:]}_{_n[-2][:3]}"

    print(f"Converting {mname}...")

    replace_cfg = {
        "blocks.0.blocks": "blocks",
        ".prenorm.": ".norm1.",
        ".norm.": ".norm2.",
    }
    expand_cfg = {"patch_embed.proj.bias": ["after", 2]}
    squeeze_cfg = {
        "pos_embed": 0,
        "cls_token": 0,
        "register_tokens": 0,
    }
    whitelist = []

    siglip2, torch_model = convert_torch_to_equinox(
        siglip2,
        replace_cfg,
        expand_cfg,
        squeeze_cfg,
        whitelist,
        strict=True,
        source="timm",
        timm_cfg=timm_cfg,
        return_torch=True,
    )

    arr = np.random.randn(3, siglip2_config["img_size"], siglip2_config["img_size"])
    jax_arr = jnp.array(arr)
    torch_arr = torch.tensor(arr).unsqueeze(0).float()

    assert (
        compare(
            jax.vmap(siglip2.norm)(siglip2.features(jax_arr, key)),
            torch_model.forward_features(torch_arr),
        )
        < 1e-5
    )

    save_model(
        Path(f"~/.cache/equimo/{mname}").expanduser(),
        siglip2,
        siglip2_config,
        timm_cfg=timm_cfg,
        compression=True,
    )

    _ = load_model(
        cls="vit",
        path=Path(f"~/.cache/equimo/{mname}.tar.lz4").expanduser(),
    )
