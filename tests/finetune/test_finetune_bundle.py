"""FineTuneBundle serialization metadata tests."""

from __future__ import annotations

from dataclasses import replace
import json
import pickle
import tarfile

import jax.random as jr
import lz4.frame
import pytest

import equimo.finetune as eqft


def test_finetune_bundle_has_required_lora_metadata(tiny_vision_transformer):
    model = eqft.apply_lora(
        tiny_vision_transformer,
        eqft.LoRAConfig(
            rank=2,
            alpha=4.0,
            target=eqft.TargetSpec(tags=("attention.proj",)),
        ),
        key=jr.PRNGKey(0),
    )

    bundle = eqft.save_delta(model, "/tmp/equimo-test-lora-bundle.eqft")

    assert bundle.method == "lora"
    assert bundle.schema_version == 1
    assert bundle.base_model_name.endswith("TinyVisionTransformer")
    assert bundle.base_model_config["dim"] == tiny_vision_transformer.dim
    assert bundle.base_checkpoint_id.startswith("sha256:")
    assert bundle.architecture_hash
    assert bundle.selector_spec["paths"] == (
        "blocks.0.attn.proj",
        "blocks.1.attn.proj",
    )
    assert bundle.trainable_labels.blocks[0].attn.proj.lora_A == "lora_A_decay"
    assert bundle.adapter_config["entries"]
    assert bundle.adapter_config["entries"][0]["path"] == "blocks.0.attn.proj"
    assert bundle.metadata["total_params"] > 0
    assert bundle.metadata["trainable_params"] > 0
    assert bundle.metadata["dtype_summary"]


def test_delta_spec_order_and_bundle_roundtrip(tmp_path, tiny_vision_transformer):
    spec = eqft.LoRAConfig(
        rank=2,
        alpha=4.0,
        target=eqft.TargetSpec(tags=("attention.proj",)),
    )
    model = eqft.apply_lora(
        tiny_vision_transformer,
        spec,
        key=jr.PRNGKey(0),
    )
    path = tmp_path / "lora.eqft"

    bundle = eqft.save_delta(path, model, tiny_vision_transformer, spec, method="lora")
    loaded_bundle = eqft.load_finetune_bundle(path)
    loaded_model = eqft.load_delta(path, tiny_vision_transformer)

    assert loaded_bundle.method == bundle.method
    assert bundle.selector_spec["target"]["tags"] == ("attention.proj",)
    assert loaded_model.blocks[0].attn.proj.lora_A.shape == model.blocks[0].attn.proj.lora_A.shape


def test_delta_spec_keyword_model_form_roundtrip(tmp_path, tiny_vision_transformer):
    spec = eqft.LoRAConfig(
        rank=2,
        alpha=4.0,
        target=eqft.TargetSpec(tags=("attention.proj",)),
    )
    model = eqft.apply_lora(
        tiny_vision_transformer,
        spec,
        key=jr.PRNGKey(0),
    )
    path = tmp_path / "lora-keyword.eqft"

    bundle = eqft.save_delta(
        path,
        model=model,
        base_model=tiny_vision_transformer,
        spec=spec,
    )
    loaded_model = eqft.load_delta(path, tiny_vision_transformer)

    assert bundle.selector_spec["target"]["tags"] == ("attention.proj",)
    assert loaded_model.blocks[0].attn.proj.lora_A.shape == model.blocks[0].attn.proj.lora_A.shape


def test_delta_rejects_same_architecture_different_base_checkpoint(
    tmp_path,
    tiny_vision_transformer,
):
    model = eqft.apply_lora(
        tiny_vision_transformer,
        eqft.LoRAConfig(
            rank=2,
            alpha=4.0,
            target=eqft.TargetSpec(tags=("attention.proj",)),
        ),
        key=jr.PRNGKey(0),
    )
    path = tmp_path / "lora.eqft"
    bundle = eqft.save_delta(model, path)
    different_base = tiny_vision_transformer.__class__(key=jr.PRNGKey(123))

    assert bundle.base_checkpoint_id.startswith("sha256:")
    with pytest.raises(eqft.FineTuneBundleError, match="base checkpoint mismatch"):
        eqft.load_delta(different_base, path)


def test_delta_file_is_pickle_free_archive(tmp_path, tiny_vision_transformer):
    model = eqft.apply_lora(
        tiny_vision_transformer,
        eqft.LoRAConfig(
            rank=2,
            alpha=4.0,
            target=eqft.TargetSpec(tags=("attention.proj",)),
        ),
        key=jr.PRNGKey(0),
    )
    path = tmp_path / "lora.eqft"

    eqft.save_delta(model, path)

    with lz4.frame.open(path, "rb") as archive:
        with tarfile.open(fileobj=archive, mode="r") as tar:
            assert set(tar.getnames()) == {"manifest.json", "arrays.eqx"}
            manifest_file = tar.extractfile("manifest.json")
            assert manifest_file is not None
            manifest = json.loads(manifest_file.read().decode())

    assert manifest["format"] == "equimo.finetune.bundle"
    assert manifest["bundle"]["method"] == "lora"
    with path.open("rb") as handle:
        with pytest.raises(pickle.UnpicklingError):
            pickle.load(handle)


def test_bundle_schema_version_is_validated(tmp_path):
    path = tmp_path / "bad.eqft"
    bundle = eqft.FineTuneBundle(method="lora", schema_version=999)

    with pytest.raises(eqft.FineTuneBundleError, match="schema_version"):
        eqft.save_finetune_bundle(path, bundle)


def test_lora_bundle_missing_path_raises_bundle_error(tmp_path, tiny_vision_transformer):
    model = eqft.apply_lora(
        tiny_vision_transformer,
        eqft.LoRAConfig(
            rank=2,
            alpha=4.0,
            target=eqft.TargetSpec(tags=("attention.proj",)),
        ),
        key=jr.PRNGKey(0),
    )
    path = tmp_path / "lora.eqft"
    bundle = eqft.save_delta(model, path)
    entries = [dict(entry) for entry in bundle.adapter_config["entries"]]
    entries[0]["path"] = "blocks.99.attn.proj"
    bad_bundle = replace(bundle, adapter_config={"entries": entries})

    with pytest.raises(eqft.FineTuneBundleError, match="no matching leaf"):
        eqft.load_delta(tiny_vision_transformer, bad_bundle)


def test_adapter_bundle_missing_path_raises_bundle_error(tmp_path, tiny_vision_transformer):
    model = eqft.apply_adapters(
        tiny_vision_transformer,
        eqft.AdapterConfig(bottleneck=3),
        key=jr.PRNGKey(0),
    )
    path = tmp_path / "adapter.eqft"
    bundle = eqft.save_delta(model, path, method="adapter")
    entries = [dict(entry) for entry in bundle.adapter_config["entries"]]
    entries[0]["path"] = "blocks.99"
    bad_bundle = replace(bundle, adapter_config={"entries": entries})

    with pytest.raises(eqft.FineTuneBundleError, match="no matching leaf"):
        eqft.load_delta(tiny_vision_transformer, bad_bundle)
