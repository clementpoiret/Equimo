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

import equimo.tabular.models as tm
from equimo.conversion.utils import stringify_name
from equimo.serialization import save_model

TABPFN_MODULE = importlib.import_module("equimo.tabular.models.tabpfn")
CHECKPOINT_DIR = Path(__file__).resolve().parent / "tabpfn3"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "data"

CHECKPOINTS = {
    "tabpfn_v3_classifier_default": {
        "path": CHECKPOINT_DIR / "tabpfn-v3-classifier-v3_default.ckpt",
        "factory": tm.tabpfn_v3_classifier_default,
        "task_type": "classification",
    },
    "tabpfn_v3_classifier_binary": {
        "path": CHECKPOINT_DIR / "tabpfn-v3-classifier-v3_20260417_binary.ckpt",
        "factory": tm.tabpfn_v3_classifier_binary,
        "task_type": "classification",
    },
    "tabpfn_v3_classifier_multiclass": {
        "path": CHECKPOINT_DIR / "tabpfn-v3-classifier-v3_20260417_multiclass.ckpt",
        "factory": tm.tabpfn_v3_classifier_multiclass,
        "task_type": "classification",
    },
    "tabpfn_v3_classifier_ood": {
        "path": CHECKPOINT_DIR / "tabpfn-v3-classifier-v3_20260506_ood.ckpt",
        "factory": tm.tabpfn_v3_classifier_ood,
        "task_type": "classification",
    },
    "tabpfn_v3_regressor_default": {
        "path": CHECKPOINT_DIR / "tabpfn-v3-regressor-v3_default.ckpt",
        "factory": tm.tabpfn_v3_regressor_default,
        "task_type": "regression",
    },
    "tabpfn_v3_regressor_mediumdata": {
        "path": CHECKPOINT_DIR / "tabpfn-v3-regressor-v3_20260417_mediumdata.ckpt",
        "factory": tm.tabpfn_v3_regressor_mediumdata,
        "task_type": "regression",
    },
    "tabpfn_v3_regressor_ood": {
        "path": CHECKPOINT_DIR / "tabpfn-v3-regressor-v3_20260506_ood.ckpt",
        "factory": tm.tabpfn_v3_regressor_ood,
        "task_type": "regression",
    },
    "tabpfn_v3_regressor_timeseries": {
        "path": CHECKPOINT_DIR / "tabpfn-v3-regressor-v3_20260506_timeseries.ckpt",
        "factory": tm.tabpfn_v3_regressor_timeseries,
        "task_type": "regression",
    },
}

REFERENCE_IDENTIFIER = "tabpfn_v3_classifier_default"
REFERENCE_PATH = OUTPUT_DIR / "tabpfn_v3_classifier_default_reference.npz"
IGNORED_TORCH_KEYS = {"regression_borders", "column_aggregator.rope.freqs"}
TOLERANCES = {"classification": 1e-4, "regression": 2e-3}


def _to_numpy(tensor):
    return np.asarray(tensor.detach().cpu().numpy(), dtype=np.float32)


def _rename_common(name: str) -> str:
    return (
        name.replace(".q_proj.", ".q_projection.")
        .replace(".k_proj.", ".k_projection.")
        .replace(".v_proj.", ".v_projection.")
        .replace(".proj.", ".out_projection.")
        .replace(".softmax_scaling.", ".softmax_scaling_layer.")
        .replace(".base_mlp.fc1.", ".base_mlp.0.")
        .replace(".base_mlp.fc2.", ".base_mlp.2.")
        .replace(".query_mlp.fc1.", ".query_mlp.0.")
        .replace(".query_mlp.fc2.", ".query_mlp.2.")
        .replace(".mlp.fc1.", ".mlp.0.")
        .replace(".mlp.fc2.", ".mlp.2.")
    )


def _map_feature_encoder(name: str) -> str | None:
    match = re.fullmatch(
        r"feature_encoder\.blocks\.blocks\.(\d+)\.inducing_vectors",
        name,
    )
    if match is not None:
        return f"feature_distribution_embedder.layers.{match.group(1)}.inducing_vectors"

    match = re.fullmatch(
        r"feature_encoder\.blocks\.blocks\.(\d+)\.attn([12])\.(.+)",
        name,
    )
    if match is None:
        return None

    layer, block, rest = match.groups()
    torch_name = _rename_common(
        f"feature_distribution_embedder.layers.{layer}.cross_attn_block{block}.{rest}"
    )
    torch_name = (
        torch_name.replace(".norm_q.", ".layernorm_q.")
        .replace(".norm_kv.", ".layernorm_kv.")
        .replace(".norm_mlp.", ".layernorm2.")
    )
    if torch_name.endswith((".layernorm_q.bias", ".layernorm_kv.bias")):
        return ""
    if torch_name.endswith(".layernorm2.bias"):
        return ""
    return torch_name


def _map_column_aggregator(name: str, readout_index: int) -> str | None:
    if name == "column_aggregator.cls_tokens":
        return "column_aggregator.cls_tokens"
    if name == "column_aggregator.norm.weight":
        return "column_aggregator.out_ln.weight"
    if name == "column_aggregator.norm.bias":
        return ""

    match = re.fullmatch(r"column_aggregator\.blocks\.blocks\.(\d+)\.(.+)", name)
    if match is not None:
        block, rest = match.groups()
        torch_name = _rename_common(f"column_aggregator.blocks.{block}.{rest}")
    elif name.startswith("column_aggregator.readout_block."):
        rest = name.split("column_aggregator.readout_block.", 1)[1]
        torch_name = _rename_common(f"column_aggregator.blocks.{readout_index}.{rest}")
    else:
        return None

    torch_name = (
        torch_name.replace(".attn.", ".attention.")
        .replace(".norm_mlp.", ".layernorm_mlp.")
        .replace(".norm.", ".layernorm.")
    )
    if torch_name.endswith((".layernorm.bias", ".layernorm_mlp.bias")):
        return ""
    return torch_name


def _map_icl_block(name: str) -> str | None:
    match = re.fullmatch(r"blocks\.0\.blocks\.(\d+)\.(.+)", name)
    if match is None:
        return None

    block, rest = match.groups()
    torch_name = _rename_common(f"icl_blocks.{block}.{rest}")
    torch_name = (
        torch_name.replace(".attn.", ".icl_attention.")
        .replace(".norm_mlp.", ".layernorm_mlp.")
        .replace(".norm.", ".layernorm.")
    )
    if torch_name.endswith((".layernorm.bias", ".layernorm_mlp.bias")):
        return ""
    return torch_name


def _array_name_for_path(name: str, *, task_type: str, readout_index: int) -> str:
    if name == "column_label_embedding.embedding.weight":
        return "col_y_encoder.embedding.weight"
    if name == "context_label_embedding.embedding.weight":
        return "icl_y_encoder.embedding.weight"
    if name == "column_label_embedding.projection.weight":
        return "col_y_encoder.weight"
    if name == "column_label_embedding.projection.bias":
        return "col_y_encoder.bias"
    if name == "context_label_embedding.projection.weight":
        return "icl_y_encoder.weight"
    if name == "context_label_embedding.projection.bias":
        return "icl_y_encoder.bias"
    if name == "norm.weight":
        return "output_norm.weight"
    if name == "norm.bias":
        return ""

    if mapped := _map_feature_encoder(name):
        return mapped
    if mapped == "":
        return ""

    mapped = _map_column_aggregator(name, readout_index)
    if mapped is not None:
        return mapped

    mapped = _map_icl_block(name)
    if mapped is not None:
        return mapped

    if task_type == "classification" and name.startswith("head."):
        return _rename_common(name.replace("head.", "many_class_decoder.", 1))
    if task_type == "regression" and name.startswith("head.fc1."):
        return name.replace("head.fc1.", "output_projection.0.", 1)
    if task_type == "regression" and name.startswith("head.fc2."):
        return name.replace("head.fc2.", "output_projection.2.", 1)

    return name


def convert_torch_to_equimo(model, torch_model, *, task_type: str):
    state = torch_model.state_dict()
    dynamic, static = eqx.partition(model, eqx.is_array)
    flat, treedef = jax.tree_util.tree_flatten_with_path(dynamic)
    readout_index = model.depths[1] - 1

    converted, used = [], set()
    for path, leaf in flat:
        name = stringify_name(path)
        torch_name = _array_name_for_path(
            name,
            task_type=task_type,
            readout_index=readout_index,
        )
        if torch_name == "":
            converted.append(leaf)
            continue
        if torch_name not in state:
            raise KeyError(f"No TabPFN conversion rule for {name!r} -> {torch_name!r}.")

        array = _to_numpy(state[torch_name])
        if tuple(array.shape) != tuple(leaf.shape):
            raise ValueError(
                f"{name}: expected shape {tuple(leaf.shape)}, got {array.shape} "
                f"from {torch_name}."
            )
        converted.append(jnp.asarray(array))
        used.add(torch_name)

    leftover = [
        key for key in state if key not in used and key not in IGNORED_TORCH_KEYS
    ]
    if leftover:
        raise KeyError(f"Unconverted Torch params: {leftover}")

    converted_tree = jax.tree_util.tree_unflatten(treedef, converted)
    return eqx.nn.inference_mode(eqx.combine(converted_tree, static), value=True)


def make_fixture(task_type: str):
    rng = np.random.default_rng(42)
    rows, columns, n_train = 12, 5, 8
    x = rng.standard_normal((rows, columns)).astype(np.float32)
    if task_type == "classification":
        y = rng.integers(0, 4, size=(rows,), dtype=np.int32)
    else:
        y = rng.standard_normal((rows,)).astype(np.float32)
    return x, y, n_train


def torch_forward(torch_model, x, y, n_train: int, task_type: str):
    import torch

    torch_x = torch.from_numpy(x).unsqueeze(1)
    if task_type == "classification":
        torch_y = torch.from_numpy(y[:n_train].astype(np.float32)).unsqueeze(1)
    else:
        torch_y = torch.from_numpy(y[:n_train]).unsqueeze(1)

    with torch.no_grad():
        out = torch_model(torch_x, torch_y)
    return _to_numpy(out.squeeze(1))


def generate_reference(torch_model, path: Path) -> None:
    x, y, n_train = make_fixture("classification")
    logits = torch_forward(torch_model, x, y, n_train, "classification")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, x=x, y=y, n_train=np.asarray(n_train), logits=logits)
    print(f"Saved TabPFN reference to {path}")


def compare(model, torch_model, *, task_type: str) -> float:
    key = jax.random.PRNGKey(42)
    x, y, n_train = make_fixture(task_type)
    torch_out = torch_forward(torch_model, x, y, n_train, task_type)
    jax_out = model(
        jnp.asarray(x),
        jnp.asarray(y),
        n_train,
        key=key,
        inference=True,
    )
    return float(np.mean(np.abs(np.asarray(jax_out) - torch_out)))


def _model_config(identifier: str) -> dict:
    base_cfg, variant_cfg = TABPFN_MODULE._TABPFN_REGISTRY[identifier]
    return base_cfg | variant_cfg


def main():
    try:
        from tabpfn.model_loading import load_model as load_tabpfn_model
    except ImportError as exc:
        raise ImportError("`torch` and `tabpfn` are required") from exc

    parser = argparse.ArgumentParser()
    parser.add_argument("identifiers", nargs="*", choices=sorted(CHECKPOINTS))
    parser.add_argument("--references-only", action="store_true")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("~/.cache/equimo/tabpfn").expanduser(),
    )
    args = parser.parse_args()

    identifiers = args.identifiers or sorted(CHECKPOINTS)
    for identifier in identifiers:
        info = CHECKPOINTS[identifier]
        print(f"Loading {info['path']}...")
        torch_model, _, _, _ = load_tabpfn_model(path=info["path"])
        torch_model.eval()

        if identifier == REFERENCE_IDENTIFIER:
            generate_reference(torch_model, REFERENCE_PATH)
        if args.references_only:
            continue

        print(f"Converting {identifier}...")
        model = info["factory"]()
        model = convert_torch_to_equimo(
            model,
            torch_model,
            task_type=info["task_type"],
        )

        error = compare(model, torch_model, task_type=info["task_type"])
        tolerance = TOLERANCES[info["task_type"]]
        assert error < tolerance, f"{identifier} conversion MAE: {error:.2e}"
        print(f"{identifier} conversion MAE: {error:.2e}")

        save_model(
            args.save_dir / identifier,
            model,
            _model_config(identifier),
            torch_hub_cfg={"checkpoint": str(info["path"])},
            compression=True,
        )


if __name__ == "__main__":
    main()
