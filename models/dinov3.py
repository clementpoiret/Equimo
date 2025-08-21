from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import equimo.models as em
from equimo.conversion.utils import convert_torch_to_equinox
from equimo.io import save_model

DIR = Path("~/.cache/torch/hub/dinov3").expanduser()


def compare(j, t) -> float:
    j = np.array(j)
    t = t.squeeze().detach().numpy()
    return float(np.mean(np.abs(j - t)))


weights = {
    # LVD
    "dinov3_vits16_pretrain_lvd1689m": str(
        (
            Path(
                "~/.cache/torch/hub/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
            )
        ).expanduser()
    ),
    "dinov3_vits16plus_pretrain_lvd1689m": str(
        (
            Path(
                "~/.cache/torch/hub/dinov3/weights/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
            )
        ).expanduser()
    ),
    "dinov3_vitb16_pretrain_lvd1689m": str(
        (
            Path(
                "~/.cache/torch/hub/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
            )
        ).expanduser()
    ),
    "dinov3_vitl16_pretrain_lvd1689m": str(
        (
            Path(
                "~/.cache/torch/hub/dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
            )
        ).expanduser()
    ),
    "dinov3_vith16plus_pretrain_lvd1689m": str(
        (
            Path(
                "~/.cache/torch/hub/dinov3/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
            )
        ).expanduser()
    ),
    "dinov3_vit7b16_pretrain_lvd1689m": str(
        (
            Path(
                "~/.cache/torch/hub/dinov3/weights/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
            )
        ).expanduser()
    ),
    # SAT
    "dinov3_vitl16_pretrain_sat493m": str(
        (
            Path(
                "~/.cache/torch/hub/dinov3/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
            )
        ).expanduser()
    ),
    "dinov3_vit7b16_pretrain_sat493m": str(
        (
            Path(
                "~/.cache/torch/hub/dinov3/weights/dinov3_vit7b16_pretrain_sat493m-a6675841.pth"
            )
        ).expanduser()
    ),
}

configs = {
    "dinov3_vits16_pretrain_lvd1689m": {
        "dim": 384,
        "num_heads": 6,
        "depths": [12],
        "reg_tokens": 4,
        "mlp_ratio": 4.0,
    },
    "dinov3_vits16plus_pretrain_lvd1689m": {
        "dim": 384,
        "num_heads": 6,
        "depths": [12],
        "reg_tokens": 4,
        "mlp_ratio": 6.0,
        "ffn_layer": "swiglu",
    },
    "dinov3_vitb16_pretrain_lvd1689m": {
        "dim": 768,
        "num_heads": 12,
        "depths": [12],
        "reg_tokens": 4,
        "mlp_ratio": 4.0,
    },
    "dinov3_vitl16_pretrain_lvd1689m": {
        "dim": 1024,
        "num_heads": 16,
        "depths": [24],
        "reg_tokens": 4,
        "mlp_ratio": 4.0,
    },
    "dinov3_vith16plus_pretrain_lvd1689m": {
        "dim": 1280,
        "num_heads": 20,
        "depths": [32],
        "reg_tokens": 4,
        "mlp_ratio": 6.0,
        "ffn_layer": "swiglu",
    },
    "dinov3_vit7b16_pretrain_lvd1689m": {
        "dim": 4096,
        "num_heads": 32,
        "depths": [40],
        "reg_tokens": 4,
        "mlp_ratio": 3.0,
        "untie_global_and_local_cls_norm": True,
        "ffn_kwargs": {"align_to": 64},
    },
    "dinov3_vitl16_pretrain_sat493m": {
        "dim": 1024,
        "num_heads": 16,
        "depths": [24],
        "reg_tokens": 4,
        "mlp_ratio": 4.0,
        "untie_global_and_local_cls_norm": True,
    },
    "dinov3_vit7b16_pretrain_sat493m": {
        "dim": 4096,
        "num_heads": 32,
        "depths": [40],
        "reg_tokens": 4,
        "mlp_ratio": 3.0,
        "untie_global_and_local_cls_norm": True,
        "ffn_kwargs": {"align_to": 64},
    },
}

citr = iter(configs.items())
name, config = next(citr)


def main():
    try:
        import torch
    except:
        raise ImportError("`torch` not available")

    key = jax.random.PRNGKey(42)
    dinov3_config = {
        "img_size": 224,
        "in_channels": 3,
        "patch_size": 16,
        "num_classes": 0,
        "use_mask_token": True,
        "use_rope_pos_embed": True,
        "reg_tokens": 4,
        "init_values": 1e-5,
        "eps": 1e-5,
        "dynamic_img_size": True,
        "act_layer": "exactgelu",
    }

    for name, config in configs.items():
        print(f"Converting {name}...")

        cfg = dinov3_config | config

        dinov3 = em.VisionTransformer(
            **cfg,
            key=key,
        )

        torch_name = "_".join(name.split("_")[:-2])
        torch_hub_cfg = {
            "repo_or_dir": str(DIR / "dinov3"),
            "model": torch_name,
            "source": "local",
            "weights": weights[name],
        }
        # model = torch.hub.load(**torch_hub_cfg)

        replace_cfg = {
            "reg_tokens": "storage_tokens",
            "blocks.0.blocks": "blocks",
            ".prenorm.": ".norm1.",
            ".norm.": ".norm2.",
        }
        expand_cfg = {"patch_embed.proj.bias": ["after", 2]}
        squeeze_cfg = {
            "pos_embed": 0,
            "cls_token": 0,
            "storage_tokens": 0,
        }
        torch_whitelist = []
        jax_whitelist = ["pos_embed.periods"]

        dinov3, torch_model = convert_torch_to_equinox(
            dinov3,
            replace_cfg,
            expand_cfg,
            squeeze_cfg,
            torch_whitelist,
            jax_whitelist,
            strict=True,
            torch_hub_cfg=torch_hub_cfg,
            return_torch=True,
        )
        dinov3 = eqx.nn.inference_mode(dinov3, True)

        arr = np.random.randn(3, cfg["img_size"], cfg["img_size"])
        jax_arr = jnp.array(arr)
        torch_arr = torch.tensor(arr).unsqueeze(0).float()

        assert (
            err := compare(
                dinov3.features(jax_arr, inference=True, key=key),
                torch_model.forward_features(torch_arr)["x_prenorm"],
            )
            < 5e-4
        ), f"Conversion error: {err}"

        save_path = Path(f"~/.cache/equimo/dinov3/{name}").expanduser()
        save_model(
            save_path,
            dinov3,
            cfg,
            torch_hub_cfg,
            compression=True,
        )

        # Ensure the serialization is okay
        # loaded_model = load_model(cls="vit", path=save_path.with_suffix(".tar.lz4"))
        # a = dinov3.features(jax_arr, inference=True, key=key)
        # b = loaded_model.features(jax_arr, inference=True, key=key)
        # jnp.mean((a - b) ** 2)


if __name__ == "__main__":
    main()
