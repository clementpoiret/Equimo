from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import torch
from einops import rearrange

import equimo.models as em
from equimo.conversion.utils import convert_torch_to_equinox
from equimo.io import save_model

DIR = Path("~/.cache/torch/hub/eupe").expanduser()
IMG_SIZE = 224


def trace_model_divergence_vit(jax_model, pt_model, arr, key):
    x_pt = torch.tensor(arr).unsqueeze(0).float()
    x_jax = jnp.array(arr)
    print(f"\n{'Module':<25} | {'Max Error':<15} | {'Mean Abs Error':<15}")
    print("-" * 62)

    # 1. Prepare Tokens
    x_pt, hw_tuple = pt_model.prepare_tokens_with_masks(x_pt)
    rope_sincos_pt = (
        pt_model.rope_embed(H=hw_tuple[0], W=hw_tuple[1])
        if getattr(pt_model, "rope_embed", None) is not None
        else None
    )

    # Jax Token Prep
    key_pos, *block_subkeys = jr.split(key, len(jax_model.blocks) + 1)
    x_jax_tokens = jax_model.patch_embed(x_jax)

    if jax_model.local_pos_embed is not None:
        if jax_model.dynamic_img_size:
            _, H, W = x_jax_tokens.shape
        else:
            H = W = jax_model.embed_size
    else:
        H = W = None

    if jax_model.global_pos_embed is not None:
        x_jax_tokens = jax_model.global_pos_embed(
            x_jax_tokens,
            cls_token=jax_model.cls_token,
            reg_tokens=jax_model.reg_tokens,
            dynamic_img_size=jax_model.dynamic_img_size,
        )
    else:
        prefix = [
            t for t in (jax_model.cls_token, jax_model.reg_tokens) if t is not None
        ]
        if jax_model.dynamic_img_size:
            x_jax_tokens = rearrange(x_jax_tokens, "c h w -> (h w) c")
        x_jax_tokens = (
            jnp.concatenate([*prefix, x_jax_tokens], axis=0) if prefix else x_jax_tokens
        )

    rope_sincos_jax = None
    if jax_model.local_pos_embed is not None:
        rope_sincos_jax = jax_model.local_pos_embed.get_sincos(
            H=H, W=W, inference=True, key=key_pos
        )

    diff = np.abs(x_pt.detach().numpy().squeeze(0) - np.array(x_jax_tokens))
    print(f"{'Token Prep':<25} | {diff.max():<15.6f} | {diff.mean():<15.6f}")
    if diff.mean() > 1e-4:
        print(f"\n[!] Divergence isolated at Token Preparation.")
        return x_pt, x_jax_tokens, pt_model.patch_embed, jax_model.patch_embed

    # 2. Evaluate Residual Blocks
    # Flatten jax blocks
    jax_blocks = []
    for chunk in jax_model.blocks:
        jax_blocks.extend(chunk.blocks)

    for j, (pt_blk, jax_blk) in enumerate(zip(pt_model.blocks, jax_blocks)):
        x_pt = (
            pt_blk(x_pt, rope_sincos_pt) if rope_sincos_pt is not None else pt_blk(x_pt)
        )
        key, subkey = jax.random.split(key)

        x_jax_tokens = jax_blk(
            x_jax_tokens, rope_sincos=rope_sincos_jax, inference=True, key=subkey
        )

        diff = np.abs(x_pt.detach().numpy().squeeze(0) - np.array(x_jax_tokens))
        print(f"Block {j:<20} | {diff.max():<15.6f} | {diff.mean():<15.6f}")
        if diff.mean() > 1e-4:
            print(f"\n[!] Divergence isolated at Block {j}.")
            return x_pt, x_jax_tokens, pt_blk, jax_blk

    print("\nNo divergence found during macro-tracing.")
    return None, None, None, None


def trace_model_divergence_convnext(jax_model, pt_model, arr, key):
    x_pt = torch.tensor(arr).unsqueeze(0).float()
    x_jax = jnp.array(arr)
    print(f"\n{'Module':<25} | {'Max Error':<15} | {'Mean Abs Error':<15}")
    print("-" * 62)
    for i in range(4):
        # 1. Evaluate Downsampler
        pt_down = pt_model.downsample_layers[i]
        jax_down = jax_model.blocks[i].downsample
        x_pt = pt_down(x_pt)
        x_jax = jax_down(x_jax, inference=True, key=jr.PRNGKey(42))
        diff = np.abs(x_pt.detach().numpy().squeeze(0) - np.array(x_jax))
        print(
            f"Stage {i} Downsampler{' ':<6} | {diff.max():<15.6f} | {diff.mean():<15.6f}"
        )
        if diff.mean() > 1e-4:
            print(f"\n[!] Divergence isolated at Stage {i} Downsampler.")
            return x_pt, x_jax, pt_down, jax_down
        # 2. Evaluate Residual Blocks
        pt_stage = pt_model.stages[i]
        jax_blocks = jax_model.blocks[i].blocks
        for j in range(len(pt_stage)):
            pt_blk = pt_stage[j]
            jax_blk = jax_blocks[j]
            x_pt = pt_blk(x_pt)
            key, subkey = jax.random.split(key)
            x_jax = jax_blk(x_jax, inference=True, key=subkey)
            diff = np.abs(x_pt.detach().numpy().squeeze(0) - np.array(x_jax))
            print(
                f"Stage {i} Block {j:<10} | {diff.max():<15.6f} | {diff.mean():<15.6f}"
            )
            if diff.mean() > 1e-4:
                print(f"\n[!] Divergence isolated at Stage {i}, Block {j}.")
                return x_pt, x_jax, pt_blk, jax_blk
    print("\nNo divergence found during macro-tracing.")
    return None, None, None, None


def _print_diff(name, pt_t, jax_t):
    diff = np.abs(pt_t.detach().numpy().squeeze(0) - np.array(jax_t))
    print(f"{name:<20} | {diff.max():<15.6f} | {diff.mean():<15.6f}")
    if diff.mean() > 1e-4:
        raise ValueError(f"Origin of divergence identified at: {name}")


def trace_intra_block(pt_blk, jax_blk, pt_input, jax_input):
    """Run this using the outputs/modules returned by the macro-tracer when it fails."""
    print(f"\n{'Operation':<20} | {'Max Error':<15} | {'Mean Abs Error':<15}")
    print("-" * 55)
    # 1. Depthwise Conv
    x_pt = pt_blk.dwconv(pt_input)
    x_jax = jax_blk.dwconv(jax_input)
    _print_diff("dwconv", x_pt, x_jax)
    # 2. Norm (Requires PT Permutation replication)
    x_pt = pt_blk.norm(x_pt.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    x_jax = jax_blk.norm(x_jax)
    _print_diff("norm", x_pt, x_jax)
    # 3. PW Conv 1
    # PT uses Linear requiring permutation; Jax uses Conv2d requiring no permutation
    x_pt = pt_blk.pwconv1(x_pt.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    x_jax = jax_blk.pwconv1(x_jax)
    _print_diff("pwconv1", x_pt, x_jax)
    # 4. Activation
    x_pt = pt_blk.act(x_pt)
    # JAX models usually apply activation functionally, fetch it from module if stored
    # If act is an Equinox module: x_jax = jax_blk.act(x_jax)
    # Otherwise, assuming standard gelu:
    x_jax = jax.nn.gelu(
        x_jax, approximate=False
    )  # Or True depending on equimo implementation
    _print_diff("act", x_pt, x_jax)
    # 5. PW Conv 2
    x_pt = pt_blk.pwconv2(x_pt.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    x_jax = jax_blk.pwconv2(x_jax)
    _print_diff("pwconv2", x_pt, x_jax)


# trace_model_divergence_vit(eupe, torch_model, arr, key)
# pt_blk = torch_model.stages[2][8]
# jax_blk = eupe.blocks[2].blocks[8]
# dummy_arr = np.random.randn(384, 14, 14).astype(np.float32)
# pt_in = torch.tensor(dummy_arr).unsqueeze(0)
# jax_in = jnp.array(dummy_arr)
# print("\nExecuting Isolated Micro-Tracer for Stage 2, Block 8...")
# trace_intra_block(pt_blk, jax_blk, pt_in, jax_in)


# Call this in your main loop instead of compare()
# pt_out, jax_out, pt_mod, jax_mod = trace_model_divergence_vit(eupe, torch_model, arr, key)


def compare(j: jax.Array, t: torch.Tensor) -> float:
    if j.ndim == 3:
        j = rearrange(j, "c h w -> (h w) c")
    j: np.ndarray = np.array(j)
    t = t.squeeze().detach().numpy()
    return float(np.mean(np.abs(j - t)))


weights = {
    # VIT
    "eupe_vitt16": str(
        (Path("~/.cache/torch/hub/eupe/weights/EUPE-ViT-T.pt")).expanduser()
    ),
    # "eupe_vits16": str(
    #     (Path("~/.cache/torch/hub/eupe/weights/EUPE-ViT-S.pt")).expanduser()
    # ),
    # "eupe_vitb16": str(
    #     (Path("~/.cache/torch/hub/eupe/weights/EUPE-ViT-B.pt")).expanduser()
    # ),
    # ConvNeXt
    # "eupe_convnext_tiny": str(
    #     (Path("~/.cache/torch/hub/eupe/weights/EUPE-ConvNeXt-T.pt")).expanduser()
    # ),
    # "eupe_convnext_small": str(
    #     (Path("~/.cache/torch/hub/eupe/weights/EUPE-ConvNeXt-S.pt")).expanduser()
    # ),
    # "eupe_convnext_base": str(
    #     (Path("~/.cache/torch/hub/eupe/weights/EUPE-ConvNeXt-B.pt")).expanduser()
    # ),
}

models = {
    "eupe_vitt16": em.eupe_vitt16,
    # "eupe_vits16": em.eupe_vits16,
    # "eupe_vitb16": em.eupe_vitb16,
    # "eupe_convnext_tiny": em.eupe_convnext_tiny,
    # "eupe_convnext_small": em.eupe_convnext_small,
    # "eupe_convnext_base": em.eupe_convnext_base,
}
convnext_sizes: dict[str, dict] = {
    "tiny": dict(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]),
    "small": dict(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768]),
    "base": dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]),
    "large": dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536]),
}


def main():
    try:
        import torch
    except:
        raise ImportError("`torch` not available")

    key = jax.random.PRNGKey(42)

    for name, path in weights.items():
        print(f"Converting {name}...")

        eupe = models[name]()
        # eupe = models[name](**config[name])

        torch_hub_cfg = {
            "repo_or_dir": str(DIR / "EUPE"),
            "model": name,
            "source": "local",
            "pretrained": True,
            "weights": path,
        }
        # model = torch.hub.load(**torch_hub_cfg)

        replace_cfg = {
            "reg_tokens": "storage_tokens",
            "blocks.0.blocks": "blocks",
            ".prenorm.": ".norm1.",
            ".norm.": ".norm2.",
        }
        # replace_cfg = {
        #     # 1. Downsample Stage 0 (Stem): Conv -> Norm
        #     "blocks.0.downsample.conv": "downsample_layers.0.0",
        #     "blocks.0.downsample.norm": "downsample_layers.0.1",
        #     # 2. Downsample Stages 1, 2, 3: Norm -> Conv
        #     "blocks.1.downsample.norm": "downsample_layers.1.0",
        #     "blocks.1.downsample.conv": "downsample_layers.1.1",
        #     "blocks.2.downsample.norm": "downsample_layers.2.0",
        #     "blocks.2.downsample.conv": "downsample_layers.2.1",
        #     "blocks.3.downsample.norm": "downsample_layers.3.0",
        #     "blocks.3.downsample.conv": "downsample_layers.3.1",
        #     # 3. Flatten the nested block paths
        #     ".blocks.": ".",
        #     # 4. Rename the main block prefixes
        #     "blocks.": "stages.",
        #     # 5. Map the layer scale gamma parameter
        #     "ls.gamma": "gamma",
        # }
        expand_cfg = {"patch_embed.proj.bias": ["after", 2]}
        # expand_cfg = {}
        squeeze_cfg = {
            "pos_embed": 0,
            "cls_token": 0,
            "storage_tokens": 0,
        }
        # squeeze_cfg = {}

        # Expand biases for the 4 downsample convolutional layers
        # expand_cfg["downsample_layers.0.0.bias"] = ["after", 2]
        # expand_cfg["downsample_layers.1.1.bias"] = ["after", 2]
        # expand_cfg["downsample_layers.2.1.bias"] = ["after", 2]
        # expand_cfg["downsample_layers.3.1.bias"] = ["after", 2]

        # Expand biases for the depthwise convolutions in all stages
        # Note: [3, 3, 9, 3] matches the ConvNeXt-Tiny layout you provided.
        # Adjust this list if you switch to Base, Large, etc.
        # for stage_idx, num_blocks in enumerate(
        #     [v["depths"] for k, v in convnext_sizes.items() if k in name][0]
        # ):
        #     for block_idx in range(num_blocks):
        #         base_path = f"stages.{stage_idx}.{block_idx}"
        #         # Depthwise conv: weight is fine (already 4D), bias needs spatial dims
        #         expand_cfg[f"{base_path}.dwconv.bias"] = ["after", 2]
        #         # Pointwise convs 1 & 2: Pt uses Linear, Jax uses Conv2d 1x1. Both weights and biases need spatial dimensions (1, 1) appended.
        #         expand_cfg[f"{base_path}.pwconv1.weight"] = ["after", 2]
        #         expand_cfg[f"{base_path}.pwconv1.bias"] = ["after", 2]
        #         expand_cfg[f"{base_path}.pwconv2.weight"] = ["after", 2]
        #         expand_cfg[f"{base_path}.pwconv2.bias"] = ["after", 2]

        torch_whitelist = []
        jax_whitelist = ["local_pos_embed.patch_rope.freqs"]

        eupe, torch_model = convert_torch_to_equinox(
            eupe,
            replace_cfg,
            expand_cfg,
            squeeze_cfg,
            torch_whitelist,
            jax_whitelist,
            strict=True,
            torch_hub_cfg=torch_hub_cfg,
            return_torch=True,
        )
        eupe = eqx.nn.inference_mode(eupe, True)

        arr = np.random.randn(3, IMG_SIZE, IMG_SIZE)
        jax_arr = jnp.array(arr)
        torch_arr = torch.tensor(arr).unsqueeze(0).float()

        trace_model_divergence_vit(eupe, torch_model, arr, key)
        assert (
            err := compare(
                eupe.features(jax_arr, inference=True, key=key),
                torch_model.forward_features(torch_arr)["x_prenorm"],
            )
        ) < 5e-4, f"Conversion error: {err}"

        print("err:", err)

        # Another eupe to compare with
        # previous_version = em.eupe_vitt16(pretrained=True)
        # previous_version = eqx.nn.inference_mode(previous_version, True)

        # Compare the new and the previous version
        # prev_out = previous_version.features(jax_arr, inference=True, key=key)
        # new_out = eupe.features(jax_arr, inference=True, key=key)

        # diff_max = float(jnp.max(jnp.abs(new_out - prev_out)))
        # diff_mean = float(jnp.mean(jnp.abs(new_out - prev_out)))
        # print(
        #     f"Comparison with previous version -> Max Error: {diff_max:.6f}, Mean Abs Error: {diff_mean:.6f}"
        # )

        # Quick comparison of the weights between the old and new versions
        # try:
        #     weight_diffs = jax.tree_util.tree_map(
        #         lambda x, y: (
        #             float(jnp.max(jnp.abs(x - y)))
        #             if eqx.is_array(x) and eqx.is_array(y) and x.shape == y.shape
        #             else 0.0
        #         ),
        #         eupe,
        #         previous_version,
        #     )
        #     max_weight_diff = max(jax.tree_util.tree_leaves(weight_diffs))
        #     print(f"Weight Comparison -> Max Error: {max_weight_diff:.6f}")
        # except Exception as e:
        #     print(f"Weight Comparison -> Tree structure mismatch: {e}")
        #     # Fallback to flattened comparison if structures differ slightly
        #     new_leaves = [x for x in jax.tree_util.tree_leaves(eupe) if eqx.is_array(x)]
        #     prev_leaves = [
        #         x
        #         for x in jax.tree_util.tree_leaves(previous_version)
        #         if eqx.is_array(x)
        #     ]
        #     diffs = [
        #         float(jnp.max(jnp.abs(n - p)))
        #         for n, p in zip(new_leaves, prev_leaves)
        #         if n.shape == p.shape
        #     ]
        #     if diffs:
        #         print(f"Weight Comparison (Flattened matching shapes) -> Max Error: {max(diffs):.6f}")

        save_path = Path(f"~/.cache/equimo/eupe/{name}").expanduser()
        save_model(
            save_path,
            eupe,
            {},
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
