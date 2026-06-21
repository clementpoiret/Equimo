"""FineTuneBundle serialization metadata tests."""

from __future__ import annotations

from dataclasses import replace
import json
import pickle
import tarfile

import jax.random as jr
import jax.numpy as jnp
import lz4.frame
import pytest

import equimo.finetune as eqft


def test_finetune_bundle_has_required_lora_metadata(tiny_vision_transformer):
    model = eqft.apply_lora(
        tiny_vision_transformer,
        eqft.LoRAConfig(
            rank=2,
            alpha=4.0,
            target=eqft.TargetSpec(tags_any=("attention.proj",)),
        ),
        key=jr.PRNGKey(0),
    )

    bundle = eqft.save_delta(model, "/tmp/equimo-test-lora-bundle.eqft")

    assert bundle.method == "lora"
    assert bundle.schema_version == 1
    assert bundle.base_model_name.endswith("TinyVisionTransformer")
    assert bundle.base_model_config["dim"] == tiny_vision_transformer.dim
    assert bundle.base_checkpoint_id.startswith("sha256:")
    assert bundle.lineage.architecture_hash == bundle.architecture_hash
    assert bundle.lineage.base_value_hash == bundle.base_checkpoint_id
    assert bundle.lineage.logical_id_table_hash
    assert bundle.lineage.model_revision == bundle.equimo_version or bundle.equimo_version == ""
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
    assert bundle.metadata["method_profile_id"] == "lora"
    assert bundle.metadata["calibration_artifact_references"] == ()
    assert bundle.metadata["recalibration_required"] is False
    assert "equimo_source_revision" in bundle.metadata


def test_delta_spec_order_and_bundle_roundtrip(tmp_path, tiny_vision_transformer):
    spec = eqft.LoRAConfig(
        rank=2,
        alpha=4.0,
        target=eqft.TargetSpec(tags_any=("attention.proj",)),
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
    assert bundle.selector_spec["target"]["tags_any"] == ("attention.proj",)
    assert loaded_model.blocks[0].attn.proj.lora_A.shape == model.blocks[0].attn.proj.lora_A.shape


def test_bundle_carries_model_state_snapshot_and_hash(tmp_path, tiny_vision_transformer):
    spec = eqft.LoRAConfig(
        rank=2,
        alpha=4.0,
        target=eqft.TargetSpec(tags_any=("attention.proj",)),
    )
    model = eqft.apply_lora(
        tiny_vision_transformer,
        spec,
        key=jr.PRNGKey(0),
    )
    model_state = {
        "batch_stats": {
            "mean": jnp.asarray([1.0, 2.0], dtype=jnp.float32),
            "var": jnp.asarray([3.0, 4.0], dtype=jnp.float32),
        },
        "updates": 7,
    }
    path = tmp_path / "stateful-lora.eqft"

    bundle = eqft.save_delta(
        path,
        model,
        tiny_vision_transformer,
        spec,
        method="lora",
        model_state=model_state,
    )
    loaded = eqft.load_finetune_bundle(path)

    assert bundle.metadata["model_state_present"] is True
    assert bundle.metadata["model_state_hash"].startswith("sha256:")
    assert bundle.metadata["recalibration_required"] is False
    assert bundle.lineage.model_state_hash == bundle.metadata["model_state_hash"]
    assert loaded.lineage.model_state_hash == bundle.lineage.model_state_hash
    assert jnp.array_equal(
        loaded.model_state["batch_stats"]["mean"],
        model_state["batch_stats"]["mean"],
    )
    assert loaded.model_state["updates"] == 7


def test_bundle_records_recalibration_marker_without_model_state(tmp_path, tiny_vision_transformer):
    spec = eqft.LoRAConfig(
        rank=2,
        alpha=4.0,
        target=eqft.TargetSpec(tags_any=("attention.proj",)),
    )
    model = eqft.apply_lora(
        tiny_vision_transformer,
        spec,
        key=jr.PRNGKey(0),
    )
    path = tmp_path / "recalibrate-lora.eqft"

    bundle = eqft.save_delta(
        path,
        model,
        tiny_vision_transformer,
        spec,
        method="lora",
        recalibration_required=True,
    )
    loaded = eqft.load_finetune_bundle(path)

    assert bundle.model_state is None
    assert bundle.metadata["model_state_present"] is False
    assert bundle.metadata["model_state_hash"] is None
    assert bundle.lineage.model_state_hash is None
    assert bundle.metadata["recalibration_required"] is True
    assert loaded.metadata["recalibration_required"] is True


def test_delta_spec_keyword_model_form_roundtrip(tmp_path, tiny_vision_transformer):
    spec = eqft.LoRAConfig(
        rank=2,
        alpha=4.0,
        target=eqft.TargetSpec(tags_any=("attention.proj",)),
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

    assert bundle.selector_spec["target"]["tags_any"] == ("attention.proj",)
    assert loaded_model.blocks[0].attn.proj.lora_A.shape == model.blocks[0].attn.proj.lora_A.shape


def test_lora_bundle_preserves_loftq_quantization_fingerprint(
    tmp_path,
    tiny_vision_transformer,
):
    config = eqft.LoftQConfig(
        rank=2,
        quantizer=eqft.QuantizerSpec(id="nf4", bits=4, format="nf4"),
    )
    quantized_weights = {
        "blocks.0.attn.proj": tiny_vision_transformer.blocks[0].attn.proj.weight
        - jnp.ones_like(tiny_vision_transformer.blocks[0].attn.proj.weight) * 0.01,
        "blocks.1.attn.proj": tiny_vision_transformer.blocks[1].attn.proj.weight
        - jnp.ones_like(tiny_vision_transformer.blocks[1].attn.proj.weight) * 0.01,
    }
    model = eqft.apply_loftq_lora(
        tiny_vision_transformer,
        config,
        quantized_weights=quantized_weights,
        target=eqft.TargetSpec(tags_any=("attention.proj",)),
        key=jr.PRNGKey(4),
    )
    path = tmp_path / "loftq.eqft"

    bundle = eqft.save_delta(model, path, method="lora")
    loaded = eqft.load_delta(tiny_vision_transformer, path)

    entry_metadata = dict(bundle.adapter_config["entries"][0]["metadata"])
    loaded_metadata = dict(loaded.blocks[0].attn.proj.metadata)
    assert entry_metadata["method"] == "loftq"
    assert entry_metadata["quantization_fingerprint"]
    assert bundle.lineage.quantization_fingerprint == bundle.metadata["quantization_fingerprint"]
    assert loaded_metadata["quantization_fingerprint"] == entry_metadata["quantization_fingerprint"]


def test_lora_bundle_rejects_tampered_quantization_lineage(
    tmp_path,
    tiny_vision_transformer,
):
    config = eqft.LoftQConfig(
        rank=2,
        quantizer=eqft.QuantizerSpec(id="nf4", bits=4, format="nf4"),
    )
    quantized_weights = {
        "blocks.0.attn.proj": tiny_vision_transformer.blocks[0].attn.proj.weight
        - jnp.ones_like(tiny_vision_transformer.blocks[0].attn.proj.weight) * 0.01,
        "blocks.1.attn.proj": tiny_vision_transformer.blocks[1].attn.proj.weight
        - jnp.ones_like(tiny_vision_transformer.blocks[1].attn.proj.weight) * 0.01,
    }
    model = eqft.apply_loftq_lora(
        tiny_vision_transformer,
        config,
        quantized_weights=quantized_weights,
        target=eqft.TargetSpec(tags_any=("attention.proj",)),
        key=jr.PRNGKey(5),
    )
    bundle = eqft.save_delta(model, tmp_path / "loftq.eqft", method="lora")
    bad_bundle = replace(
        bundle,
        lineage=replace(bundle.lineage, quantization_fingerprint="sha256:bad"),
    )

    with pytest.raises(eqft.FineTuneBundleError, match="quantization fingerprint mismatch"):
        eqft.load_delta(tiny_vision_transformer, bad_bundle)


def test_delta_rejects_same_architecture_different_base_checkpoint(
    tmp_path,
    tiny_vision_transformer,
):
    model = eqft.apply_lora(
        tiny_vision_transformer,
        eqft.LoRAConfig(
            rank=2,
            alpha=4.0,
            target=eqft.TargetSpec(tags_any=("attention.proj",)),
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
            target=eqft.TargetSpec(tags_any=("attention.proj",)),
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
            target=eqft.TargetSpec(tags_any=("attention.proj",)),
        ),
        key=jr.PRNGKey(0),
    )
    path = tmp_path / "lora.eqft"
    bundle = eqft.save_delta(model, path)
    entries = [dict(entry) for entry in bundle.adapter_config["entries"]]
    entries[0]["path"] = "blocks.99.attn.proj"
    bad_bundle = replace(bundle, adapter_config={"entries": entries})

    with pytest.raises(eqft.FineTuneBundleError, match="logical-ID table mismatch|no matching leaf"):
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

    with pytest.raises(eqft.FineTuneBundleError, match="logical-ID table mismatch|no matching leaf"):
        eqft.load_delta(tiny_vision_transformer, bad_bundle)
