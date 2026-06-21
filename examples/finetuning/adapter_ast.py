"""Adapter example using a tiny local transformer-like model."""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import equimo.finetune as eqft


class TinyBlock(eqx.Module):
    mlp: eqx.nn.Linear

    def __init__(self, dim: int, *, key):
        self.mlp = eqx.nn.Linear(dim, dim, key=key)

    def __call__(self, x, **kwargs):
        del kwargs
        return self.mlp(x)


class TinyASTLike(eqx.Module):
    blocks: tuple[TinyBlock, ...]
    head: eqx.nn.Linear

    def __init__(self, *, key):
        key0, key1, key_head = jr.split(key, 3)
        self.blocks = (
            TinyBlock(4, key=key0),
            TinyBlock(4, key=key1),
        )
        self.head = eqx.nn.Linear(4, 2, key=key_head)

    def __call__(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(x)


model = TinyASTLike(key=jr.PRNGKey(0))
adapted = eqft.apply_adapters(
    model,
    eqft.AdapterConfig(bottleneck=2),
    key=jr.PRNGKey(1),
)
plan = eqft.prepare_finetune(
    adapted,
    trainable=eqft.TrainableSpec(
        mode="peft",
        method_name="adapter",
        train_head=True,
    ),
)

print(adapted(jnp.ones((4,))).shape)
print(plan.report)
