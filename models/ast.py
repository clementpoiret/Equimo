import argparse
import importlib
import re
import sys
from pathlib import Path

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    if sys.path and Path(sys.path[0]).resolve() == script_dir:
        sys.path.pop(0)

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import equimo.audio.models as am
from equimo.conversion.utils import stringify_name
from equimo.serialization import save_model

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "data"
AST_MODULE = importlib.import_module("equimo.audio.models.ast")

CHECKPOINTS = {
    "ast_base_patch16_audioset_10_10_0_4593": {
        "hf_id": "MIT/ast-finetuned-audioset-10-10-0.4593",
        "factory": am.ast_base_patch16_audioset_10_10_0_4593,
        "reference": OUTPUT_DIR
        / "ast_base_patch16_audioset_10_10_0_4593_reference.npz",
        "input_shape": (1024, 128),
    },
    "ast_base_patch16_speechcommands_v2_10_10_0_9812": {
        "hf_id": "MIT/ast-finetuned-speech-commands-v2",
        "factory": am.ast_base_patch16_speechcommands_v2_10_10_0_9812,
        "reference": OUTPUT_DIR
        / "ast_base_patch16_speechcommands_v2_10_10_0_9812_reference.npz",
        "input_shape": (128, 128),
    },
}


def _to_numpy(tensor):
    return np.asarray(tensor.detach().cpu().numpy(), dtype=np.float32)


def _state_array(state, name):
    if name not in state:
        raise KeyError(f"{name!r} not found in Hugging Face AST state dict.")
    return _to_numpy(state[name])


def _block_array(state, block_index: str, leaf: str):
    prefix = f"audio_spectrogram_transformer.layers.{block_index}"

    if leaf in ("attn.qkv.weight", "attn.qkv.bias"):
        suffix = "weight" if leaf.endswith("weight") else "bias"
        return np.concatenate(
            [
                _state_array(state, f"{prefix}.attention.q_proj.{suffix}"),
                _state_array(state, f"{prefix}.attention.k_proj.{suffix}"),
                _state_array(state, f"{prefix}.attention.v_proj.{suffix}"),
            ],
            axis=0,
        )

    leaf_map = {
        "prenorm.weight": "layernorm_before.weight",
        "prenorm.bias": "layernorm_before.bias",
        "norm.weight": "layernorm_after.weight",
        "norm.bias": "layernorm_after.bias",
        "attn.proj.weight": "attention.o_proj.weight",
        "attn.proj.bias": "attention.o_proj.bias",
        "mlp.fc1.weight": "mlp.fc1.weight",
        "mlp.fc1.bias": "mlp.fc1.bias",
        "mlp.fc2.weight": "mlp.fc2.weight",
        "mlp.fc2.bias": "mlp.fc2.bias",
    }
    return _state_array(state, f"{prefix}.{leaf_map[leaf]}")


def _array_for_path(state, path: str):
    if path == "patch_embed.proj.weight":
        return _state_array(
            state,
            "audio_spectrogram_transformer.embeddings.patch_embeddings.projection.weight",
        )
    if path == "patch_embed.proj.bias":
        return _state_array(
            state,
            "audio_spectrogram_transformer.embeddings.patch_embeddings.projection.bias",
        )[:, None, None]
    if path == "pos_embed":
        return _state_array(
            state,
            "audio_spectrogram_transformer.embeddings.position_embeddings",
        )[0]
    if path == "cls_token":
        return _state_array(
            state,
            "audio_spectrogram_transformer.embeddings.cls_token",
        )[0]
    if path == "dist_token":
        return _state_array(
            state,
            "audio_spectrogram_transformer.embeddings.distillation_token",
        )[0]
    if path in ("norm.weight", "norm.bias"):
        suffix = path.rsplit(".", 1)[1]
        return _state_array(state, f"audio_spectrogram_transformer.layernorm.{suffix}")
    if path in ("head_norm.weight", "head_norm.bias"):
        suffix = path.rsplit(".", 1)[1]
        return _state_array(state, f"classifier.layernorm.{suffix}")
    if path in ("head.weight", "head.bias"):
        suffix = path.rsplit(".", 1)[1]
        return _state_array(state, f"classifier.dense.{suffix}")

    match = re.fullmatch(r"blocks\.0\.blocks\.(\d+)\.(.+)", path)
    if match is not None:
        return _block_array(state, match.group(1), match.group(2))

    raise KeyError(f"No AST conversion rule for Equimo parameter {path!r}.")


def convert_hf_to_equimo(model, hf_model):
    state = hf_model.state_dict()
    dynamic, static = eqx.partition(model, eqx.is_array)
    flat, treedef = jax.tree_util.tree_flatten_with_path(dynamic)

    converted = []
    for path, leaf in flat:
        name = stringify_name(path)
        array = _array_for_path(state, name)
        if tuple(array.shape) != tuple(leaf.shape):
            raise ValueError(f"{name}: expected shape {leaf.shape}, got {array.shape}.")
        converted.append(jnp.asarray(array))

    converted_tree = jax.tree_util.tree_unflatten(treedef, converted)
    return eqx.nn.inference_mode(eqx.combine(converted_tree, static), value=True)


def compare(jax_array, torch_tensor) -> float:
    return float(np.mean(np.abs(np.array(jax_array) - _to_numpy(torch_tensor))))


def generate_reference(hf_model, input_values, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch_x = torch.from_numpy(input_values).unsqueeze(0)

    with torch.no_grad():
        features = hf_model.audio_spectrogram_transformer(
            input_values=torch_x
        ).last_hidden_state
        logits = hf_model(input_values=torch_x).logits

    np.savez(
        path,
        input_values=input_values,
        cls_token=_to_numpy(features[0, 0]),
        dist_token=_to_numpy(features[0, 1]),
        logits=_to_numpy(logits[0]),
    )
    print(f"Saved AST reference to {path}")


def main():
    try:
        import torch
        from transformers import ASTForAudioClassification
    except ImportError as exc:
        raise ImportError("`torch` and `transformers` are required") from exc

    parser = argparse.ArgumentParser()
    parser.add_argument("identifiers", nargs="*", choices=sorted(CHECKPOINTS))
    parser.add_argument("--references-only", action="store_true")
    args = parser.parse_args()

    key = jax.random.PRNGKey(42)
    identifiers = args.identifiers or sorted(CHECKPOINTS)

    for identifier in identifiers:
        rng = np.random.default_rng(42)
        info = CHECKPOINTS[identifier]
        print(f"Loading {info['hf_id']}...")
        hf_model = ASTForAudioClassification.from_pretrained(info["hf_id"]).eval()
        input_values = rng.standard_normal(info["input_shape"]).astype(np.float32)
        generate_reference(hf_model, input_values, info["reference"])

        if args.references_only:
            continue

        print(f"Converting {identifier}...")
        model = info["factory"]()
        model = convert_hf_to_equimo(model, hf_model)

        jax_x = jnp.asarray(input_values)
        torch_x = torch.from_numpy(input_values).unsqueeze(0)
        with torch.no_grad():
            hf_features = hf_model.audio_spectrogram_transformer(
                input_values=torch_x
            ).last_hidden_state
            hf_logits = hf_model(input_values=torch_x).logits

        fwd = model.forward_features(jax_x, key=key, inference=True)
        cls_error = compare(fwd["x_norm_cls_token"], hf_features[0, 0])
        dist_error = compare(fwd["x_norm_dist_token"], hf_features[0, 1])
        logits_error = compare(model(jax_x, key=key, inference=True), hf_logits[0])
        assert cls_error < 5e-4, f"cls token error: {cls_error:.2e}"
        assert dist_error < 5e-4, f"dist token error: {dist_error:.2e}"
        assert logits_error < 5e-4, f"logits error: {logits_error:.2e}"

        base_cfg, variant_cfg = AST_MODULE._AST_REGISTRY[identifier]
        cfg = base_cfg | variant_cfg
        save_model(
            Path(f"~/.cache/equimo/ast/{identifier}").expanduser(),
            model,
            cfg,
            torch_hub_cfg={"hf_model": info["hf_id"]},
            compression=True,
        )


if __name__ == "__main__":
    main()
