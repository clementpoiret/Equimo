"""Cross-method PEFT invariant smoke tests."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

import equimo.finetune as eqft


def _assert_filter_jit_and_vmap(model):
    x = jnp.ones((2, 3), dtype=jnp.float32)

    y_jit = eqx.filter_jit(model)(x)
    y_batch = jax.vmap(model)(jnp.stack([x, x]))

    assert y_jit.shape == (2,)
    assert y_batch.shape == (2, 2)


def test_peft_wrappers_filter_jit_and_vmap(tiny_vision_transformer):
    key = jr.PRNGKey(0)
    methods = (
        eqft.apply_lora(
            tiny_vision_transformer,
            eqft.LoRAConfig(
                rank=2,
                target=eqft.TargetSpec(tags_any=("attention.proj",)),
            ),
            key=key,
        ),
        eqft.apply_dora(
            tiny_vision_transformer,
            eqft.DoRAConfig(
                rank=2,
                target=eqft.TargetSpec(tags_any=("attention.proj",)),
            ),
            key=key,
        ),
        eqft.apply_adapters(
            tiny_vision_transformer,
            eqft.AdapterConfig(bottleneck=2),
            key=key,
        ),
        eqft.apply_prompts(
            tiny_vision_transformer,
            eqft.PromptConfig(num_tokens=2),
            key=key,
        ),
        eqft.apply_prefixes(
            tiny_vision_transformer,
            eqft.PrefixConfig(num_prefix_tokens=2),
            key=key,
        ),
        eqft.apply_scale_shift(
            tiny_vision_transformer,
            eqft.ScaleShiftConfig(target=eqft.TargetSpec(include=("*.norm",))),
        ),
        eqft.apply_ia3(
            tiny_vision_transformer,
            eqft.IA3Config(target=eqft.TargetSpec(tags_any=("attention.proj",))),
        ),
        eqft.apply_vera(
            tiny_vision_transformer,
            eqft.VeRAConfig(
                rank=3,
                target=eqft.TargetSpec(tags_any=("attention.proj",)),
            ),
            key=key,
        ),
    )

    for model in methods:
        _assert_filter_jit_and_vmap(model)
