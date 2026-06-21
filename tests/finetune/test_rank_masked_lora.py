"""Static rank-masked LoRA tests."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

import equimo.finetune as eqft


class TinyLinearModel(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self):
        self.linear = eqx.nn.Linear(3, 2, key=jr.PRNGKey(0))

    def __call__(self, x):
        return self.linear(x)


def test_rank_masked_lora_delta_roundtrip(tmp_path, tiny_vision_transformer):
    model = eqft.apply_lora(
        tiny_vision_transformer,
        eqft.RankMaskedLoRAConfig(
            rank=4,
            target_rank=2,
            rank_mask_init="target_rank",
            target=eqft.TargetSpec(tags=("attention.proj",)),
        ),
        key=jr.PRNGKey(0),
    )
    path = tmp_path / "rank_mask.eqft"

    eqft.save_delta(model, path)
    loaded = eqft.load_delta(tiny_vision_transformer, path)

    assert jnp.array_equal(
        loaded.blocks[0].attn.proj.rank_mask,
        jnp.array([1, 1, 0, 0], dtype=loaded.blocks[0].attn.proj.rank_mask.dtype),
    )


def test_rank_masked_lora_default_starts_all_initial_ranks_active(tiny_vision_transformer):
    model = eqft.apply_lora(
        tiny_vision_transformer,
        eqft.RankMaskedLoRAConfig(
            rank=4,
            initial_rank=4,
            target_rank=2,
            target=eqft.TargetSpec(tags=("attention.proj",)),
        ),
        key=jr.PRNGKey(0),
    )
    plan = eqft.prepare_finetune(
        model,
        trainable=eqft.TrainableSpec(mode="peft", method_name="lora", train_head=False),
    )

    assert jnp.array_equal(
        model.blocks[0].attn.proj.rank_mask,
        jnp.array([1, 1, 1, 1], dtype=model.blocks[0].attn.proj.rank_mask.dtype),
    )
    assert plan.trainable.blocks[0].attn.proj.rank_mask is None


def test_lora_rank_groups_use_canonical_path_strings():
    model = eqft.apply_lora(
        TinyLinearModel(),
        eqft.RankMaskedLoRAConfig(
            rank=4,
            initial_rank=4,
            target_rank=2,
            target=eqft.TargetSpec(predicate=eqft.is_linear),
        ),
        key=jr.PRNGKey(0),
    )

    assert eqft.lora_rank_groups(model) == {"linear": 4}


def test_apply_lora_rank_pattern_updates_rank_masks():
    model = eqft.apply_lora(
        TinyLinearModel(),
        eqft.RankMaskedLoRAConfig(
            rank=4,
            initial_rank=4,
            target_rank=2,
            target=eqft.TargetSpec(predicate=eqft.is_linear),
        ),
        key=jr.PRNGKey(0),
    )
    mask = jnp.array([True, False, True, False])

    updated = eqft.apply_lora_rank_pattern(model, {"linear": mask})

    assert jnp.array_equal(updated.linear.rank_mask, mask)


def test_apply_lora_rank_pattern_validates_paths_and_shapes():
    model = eqft.apply_lora(
        TinyLinearModel(),
        eqft.RankMaskedLoRAConfig(
            rank=4,
            initial_rank=4,
            target_rank=2,
            target=eqft.TargetSpec(predicate=eqft.is_linear),
        ),
        key=jr.PRNGKey(0),
    )

    with pytest.raises(ValueError, match="unknown LoRA module paths"):
        eqft.apply_lora_rank_pattern(
            model,
            {"missing": jnp.ones((4,), dtype=jnp.bool_)},
        )

    with pytest.raises(ValueError, match=r"shape \(4,\)"):
        eqft.apply_lora_rank_pattern(
            model,
            {"linear": jnp.ones((3,), dtype=jnp.bool_)},
        )


def test_apply_lora_rank_pattern_rejects_merged_modules():
    model = eqft.apply_lora(
        TinyLinearModel(),
        eqft.RankMaskedLoRAConfig(
            rank=4,
            initial_rank=4,
            target_rank=2,
            target=eqft.TargetSpec(predicate=eqft.is_linear),
        ),
        key=jr.PRNGKey(0),
    )
    merged = eqft.merge_lora(model)

    with pytest.raises(ValueError, match="merged LoRA module"):
        eqft.apply_lora_rank_pattern(
            merged,
            {"linear": jnp.ones((4,), dtype=jnp.bool_)},
        )


def test_rank_mask_affects_unmerged_lora_forward():
    model = eqft.apply_lora(
        TinyLinearModel(),
        eqft.RankMaskedLoRAConfig(
            rank=2,
            initial_rank=2,
            target_rank=1,
            target=eqft.TargetSpec(predicate=eqft.is_linear),
        ),
        key=jr.PRNGKey(0),
    )
    lora_A = jnp.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=model.linear.lora_A.dtype,
    )
    lora_B = jnp.array(
        [[1.0, 10.0], [2.0, 20.0]],
        dtype=model.linear.lora_B.dtype,
    )
    model = eqx.tree_at(
        lambda tree: (tree.linear.lora_A, tree.linear.lora_B),
        model,
        (lora_A, lora_B),
    )
    masked = eqft.apply_lora_rank_pattern(
        model,
        {"linear": jnp.array([True, False])},
    )
    x = jnp.array([1.0, 1.0, 0.0])

    expected = masked.linear.base(x) + masked.linear.delta_weight() @ x

    assert not jnp.allclose(model(x), masked(x))
    assert jnp.allclose(masked(x), expected)
