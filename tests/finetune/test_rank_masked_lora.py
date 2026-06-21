"""Static rank-masked LoRA tests."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

import equimo.finetune as eqft


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
