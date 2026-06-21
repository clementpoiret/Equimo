"""Prompt tuning tests."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

import equimo.finetune as eqft
from equimo.vision.models.vit import VisionTransformer

from fixtures import TinyVisionTransformer


class MixingBlock(eqx.Module):
    def __call__(self, x, **kwargs):
        del kwargs
        return x + jnp.mean(x, axis=0)


def test_vpt_deep_shapes(tiny_vision_transformer):
    prompted = eqft.apply_prompts(
        tiny_vision_transformer,
        eqft.VPTDeepConfig(num_tokens=3),
        key=jr.PRNGKey(0),
    )
    x = jnp.ones((2, 3))

    features = prompted.features(x)
    logits = prompted(x)

    assert features.shape == (6, 4)
    assert logits.shape == (2,)


def test_vpt_deep_inserts_before_replacing_prompt_slots():
    model = TinyVisionTransformer(depth=2, num_reg_tokens=2)
    prompted = eqft.apply_prompts(
        model,
        eqft.VPTDeepConfig(num_tokens=2),
        key=jr.PRNGKey(0),
    )
    x = jnp.ones((2, 3))

    features = prompted.features(x)

    assert features.shape == (7, 4)


def test_vpt_preserves_register_tokens_before_blocks():
    model = TinyVisionTransformer(depth=0, num_reg_tokens=2)
    reg_tokens = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)
    model = eqx.tree_at(lambda m: m.reg_tokens, model, reg_tokens)
    model = eqx.tree_at(lambda m: m.norm, model, eqx.nn.Identity())
    prompted = eqft.apply_prompts(
        model,
        eqft.VPTDeepConfig(num_tokens=2),
        key=jr.PRNGKey(0),
    )
    x = jnp.ones((2, 3))

    features = prompted.features(x)

    assert features.shape == (7, 4)
    assert jnp.array_equal(features[3:5], reg_tokens)


def test_vpt_pooling_excludes_prompts(tiny_vision_transformer):
    prompted = eqft.apply_prompts(
        tiny_vision_transformer,
        eqft.PromptConfig(num_tokens=3),
        key=jr.PRNGKey(0),
    )
    x = jnp.ones((2, 3))

    pooled = eqft.extract_features(prompted, x, pool="mean_patch")
    features = prompted.features(x)
    manual = jnp.mean(features[4:], axis=0)

    assert jnp.allclose(pooled, manual, atol=1e-6)


def test_vpt_real_vit_register_rope_shape():
    model = VisionTransformer(
        img_size=32,
        in_channels=3,
        patch_size=16,
        dim=8,
        num_heads=2,
        depths=[1],
        reg_tokens=4,
        num_classes=3,
        use_global_pos_embed=False,
        use_local_pos_embed=True,
        local_pos_embed_reg=True,
        key=jr.PRNGKey(0),
    )
    prompted = eqft.apply_prompts(
        model,
        eqft.VPTDeepConfig(num_tokens=2),
        key=jr.PRNGKey(1),
    )
    x = jnp.ones((3, 32, 32))

    features = prompted.features(x, key=jr.PRNGKey(2), inference=True)

    assert features.shape == (11, 8)


def test_prompt_trainable_only_prompts_and_head(tiny_vision_transformer):
    prompted = eqft.apply_prompts(
        tiny_vision_transformer,
        eqft.PromptConfig(num_tokens=3),
        key=jr.PRNGKey(0),
    )
    plan = eqft.prepare_finetune(
        prompted,
        trainable=eqft.TrainableSpec(mode="peft", method_name="prompt", train_head=True),
    )

    assert plan.trainable.prompts[0] is not None
    assert plan.trainable.base.patch_embed.proj.weight is None
    assert plan.trainable.base.head.weight is not None
    assert "prompt_decay" in plan.report.trainable_by_label


def test_soft_prompt_uses_text_embedding_initialization(tiny_text_encoder):
    prompted = eqft.apply_prompts(
        tiny_text_encoder,
        eqft.SoftPromptConfig(num_tokens=2),
        key=jr.PRNGKey(0),
    )
    token_ids = jnp.array([0, 1, 2])

    features = prompted.features(token_ids)
    logits = prompted(token_ids)
    plan = eqft.prepare_finetune(
        prompted,
        trainable=eqft.TrainableSpec(
            mode="peft",
            method_name="prompt",
            train_head=False,
        ),
    )

    assert jnp.array_equal(prompted.prompts[0], tiny_text_encoder.token_embed.weight[:2])
    assert features.shape == (5, tiny_text_encoder.dim)
    assert logits.shape == (tiny_text_encoder.token_embed.weight.shape[0],)
    assert "prompt_decay" in plan.report.trainable_by_label
    assert plan.trainable.base.head.weight is None


def test_deep_prompt_config_can_share_across_layers():
    model = TinyVisionTransformer(depth=2)
    prompted = eqft.apply_prompts(
        model,
        eqft.PTuningV2Config(num_tokens=2, share_across_layers=True),
        key=jr.PRNGKey(0),
    )

    features = prompted.features(jnp.ones((2, 3)))

    assert len(prompted.prompts) == 1
    assert features.shape == (5, model.dim)


def test_ptuning_v2_rejects_unimplemented_mlp_reparameterizer(
    tiny_text_encoder,
):
    with pytest.raises(ValueError, match="reparameterizer"):
        eqft.apply_prompts(
            tiny_text_encoder,
            eqft.PTuningV2Config(reparameterizer="mlp"),
            key=jr.PRNGKey(0),
        )


def test_prompts_receive_gradients_when_blocks_mix_tokens(tiny_vision_transformer):
    model = eqx.tree_at(
        lambda m: m.blocks,
        tiny_vision_transformer,
        tuple(MixingBlock() for _ in tiny_vision_transformer.blocks),
    )
    prompted = eqft.apply_prompts(
        model,
        eqft.VPTDeepConfig(num_tokens=2),
        key=jr.PRNGKey(0),
    )
    x = jnp.ones((2, 3))

    def loss_fn(prompts):
        local = eqx.tree_at(lambda m: m.prompts, prompted, prompts)
        return jnp.sum(local(x))

    grads = eqx.filter_grad(loss_fn)(prompted.prompts)

    assert all(jnp.any(grad != 0) for grad in grads)
