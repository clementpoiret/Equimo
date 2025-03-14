from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from equimo.experimental.text import TextEncoder
from equimo.conversion.utils import convert_torch_to_equinox
from equimo.io import save_model
from tips import text_encoder

CKPT_PATHS = {
    "tips_vits14_hr_text": "/mnt/hdd/torch/tips/tips_oss_s14_highres_distilled_text.npz",
    "tips_vitb14_hr_text": "/mnt/hdd/torch/tips/tips_oss_b14_highres_distilled_text.npz",
    "tips_vitl14_hr_text": "/mnt/hdd/torch/tips/tips_oss_l14_highres_distilled_text.npz",
    "tips_vitso400m14_hr_text": "/mnt/hdd/torch/tips/tips_oss_so400m14_highres_largetext_distilled_text.npz",
    "tips_vitg14_hr_text": "/mnt/hdd/torch/tips/tips_oss_g14_highres_text.npz",
    "tips_vitg14_lr_text": "/mnt/hdd/torch/tips/tips_oss_g14_lowres_text.npz",
}


def compare(j, t) -> float:
    j = np.array(j)
    t = t.squeeze().detach().numpy()
    return float(np.mean(np.abs(j - t)))


configs = {
    "tips_vits14_hr_text": {
        "dim": 384,
        "num_heads": 6,
        "depth": 12,
        "mlp_ratio": 4.0,
        "temperature": 0.005497702397406101,
    },
    "tips_vitb14_hr_text": {
        "dim": 768,
        "num_heads": 12,
        "depth": 12,
        "mlp_ratio": 4.0,
        "temperature": 0.00397537462413311,
    },
    "tips_vitl14_hr_text": {
        "dim": 1024,
        "num_heads": 16,
        "depth": 12,
        "mlp_ratio": 4.0,
        "temperature": 0.004205586854368448,
    },
    "tips_vitso400m14_hr_text": {
        "dim": 1152,
        "num_heads": 16,
        "depth": 27,
        "mlp_ratio": 4304 / 1152,
        "temperature": 0.002699660835787654,
    },
    "tips_vitg14_hr_text": {
        "dim": 1536,
        "num_heads": 24,
        "depth": 12,
        "mlp_ratio": 4.0,
        "temperature": 0.003517505945637822,
    },
    "tips_vitg14_lr_text": {
        "dim": 1536,
        "num_heads": 24,
        "depth": 12,
        "mlp_ratio": 4.0,
        "temperature": 0.003806645981967449,
    },
}


def get_text_config(v):
    return {
        "hidden_size": {
            "vits14": 384,
            "vitb14": 768,
            "vitl14": 1024,
            "vitso400m14": 1152,
            "vitg14": 1536,
        }[v],
        "mlp_dim": {
            "vits14": 1536,
            "vitb14": 3072,
            "vitl14": 4096,
            "vitso400m14": 4304,
            "vitg14": 6144,
        }[v],
        "num_heads": {
            "vits14": 6,
            "vitb14": 12,
            "vitl14": 16,
            "vitso400m14": 16,
            "vitg14": 24,
        }[v],
        "num_layers": {
            "vits14": 12,
            "vitb14": 12,
            "vitl14": 12,
            "vitso400m14": 27,
            "vitg14": 12,
        }[v],
    }


# citr = iter(configs.items())
# name, config = next(citr)


def main():
    try:
        import torch
    except:
        raise ImportError("`torch` not available")

    key = jax.random.PRNGKey(42)
    base_config = {
        "vocab_size": 32000,
        "scale_sqrt_depth": True,
        "act_layer": "relu",
    }

    for name, config in configs.items():
        print(f"Converting {name}...")

        cfg = base_config | config

        tips_text = TextEncoder(
            **cfg,
            key=key,
        )

        weights_text = dict(np.load(CKPT_PATHS[name], allow_pickle=False))
        for k in weights_text:
            weights_text[k] = torch.tensor(weights_text[k])

        with torch.no_grad():
            model_text = text_encoder.TextEncoder(
                get_text_config(name.split("_")[1]),
                vocab_size=32000,
            )
            temperature = weights_text.pop("temperature")

            assert cfg["temperature"] == temperature, (
                f"There is a temp mismatch. Got {cfg['temperature']}, expected {temperature}."
            )

            model_text.load_state_dict(weights_text)

        replace_cfg = {
            "reg_tokens": "register_tokens",
            "blocks.": "resblocks.",
            ".prenorm.": ".ln_1.",
            ".norm.": ".ln_2.",
            ".qkv.weight": ".in_proj_weight",
            ".qkv.bias": ".in_proj_bias",
            ".attn.proj.": ".attn.out_proj.",
            ".mlp.fc1.": ".mlp.c_fc.",
            ".mlp.fc2.": ".mlp.c_proj.",
        }
        expand_cfg = {"patch_embed.proj.bias": ["after", 2]}
        squeeze_cfg = {
            "pos_embed": 0,
            "cls_token": 0,
            "register_tokens": 0,
        }
        whitelist = []

        tips_text, torch_model = convert_torch_to_equinox(
            tips_text,
            replace_cfg,
            expand_cfg,
            squeeze_cfg,
            whitelist,
            strict=True,
            source="custom",
            torch_model=model_text,
            return_torch=True,
        )

        ids = np.random.randint(0, 100, size=(64))
        paddings = np.zeros_like(ids)
        jax_ids = jnp.array(ids)
        jax_paddings = jnp.array(paddings)

        torch_ids = torch.from_numpy(ids).unsqueeze(0)
        torch_paddings = torch.from_numpy(paddings).unsqueeze(0)

        assert (
            compare(
                tips_text(jax_ids, jax_paddings, key=key),
                torch_model(torch_ids, torch_paddings),
            )
            < 1e-5
        )

        save_model(
            Path(f"~/.cache/equimo/tips/{name}").expanduser(),
            tips_text,
            cfg,
            compression=True,
        )
