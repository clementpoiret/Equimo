"""Tests for equimo.layers.attention."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
import equinox as eqx

from equimo.layers.attention import (
    Attention,
    WindowedAttention,
    AttentionBlock,
    HATBlock,
    SHSA,
    SHMA,
    SHMABlock,
    LinearAttention,
    MllaBlock,
    MMSA,
    SQA,
    PartialFormerBlock,
    LinearAngularAttention,
    RFAttention,
    RFAttentionBlock,
    ConvAttention,
    ConvAttentionBlock,
    LowFormerBlock,
    get_attn,
    get_attn_block,
)

KEY = jr.PRNGKey(0)
DIM = 32
NUM_HEADS = 4
SEQLEN = 16
H, W = 4, 4


class TestAttentionLayers:
    @pytest.mark.parametrize(
        "cls, kwargs",
        [
            (Attention, {"dim": DIM, "num_heads": NUM_HEADS}),
            (WindowedAttention, {"dim": DIM, "num_heads": NUM_HEADS, "resolution": 4, "seq_len": 16}),
            (SHSA, {"dim": DIM, "qk_dim": 8, "pdim": 8}),
            (SHMA, {"dim": DIM, "num_heads": 1}),
            (LinearAttention, {"input_resolution": (4, 4), "dim": DIM, "num_heads": NUM_HEADS}),
            (MMSA, {"dim": DIM, "num_heads": NUM_HEADS}),
            (LinearAngularAttention, {"dim": DIM, "num_heads": NUM_HEADS}),
            (RFAttention, {"in_channels": DIM, "out_channels": DIM}),
            (ConvAttention, {"in_channels": DIM}),
        ],
    )
    def test_attention_forward(self, cls, kwargs):
        key = KEY
        model = cls(**kwargs, key=key)
        
        if cls in (SHSA, SHMA, RFAttention, ConvAttention):
            x = jr.normal(key, (DIM, H, W))
        else:
            x = jr.normal(key, (SEQLEN, DIM))
            
        out = model(x, key=key, inference=True)
        assert out.shape == x.shape
        assert jnp.all(jnp.isfinite(out))

    @pytest.mark.parametrize(
        "cls, kwargs",
        [
            (AttentionBlock, {"dim": DIM, "num_heads": NUM_HEADS}),
            (HATBlock, {"dim": DIM, "num_heads": NUM_HEADS, "window_size": 4, "sr_ratio": 2}),
            (SHMABlock, {"dim": DIM}),
            (MllaBlock, {"dim": DIM, "input_resolution": (4, 4), "num_heads": NUM_HEADS}),
            (PartialFormerBlock, {"dim": DIM, "num_heads": NUM_HEADS, "foreground_ratio": 0.5, "patch_size": 2}),
            (RFAttentionBlock, {"in_channels": DIM}),
            (ConvAttentionBlock, {"dim": DIM}),
            (LowFormerBlock, {"dim": DIM}),
        ],
    )
    def test_block_forward(self, cls, kwargs):
        key = KEY
        model = cls(**kwargs, key=key)
        
        if cls in (SHMABlock, RFAttentionBlock, LowFormerBlock, ConvAttentionBlock):
            x = jr.normal(key, (DIM, H, W))
        else:
            x = jr.normal(key, (SEQLEN, DIM))

        if cls == HATBlock:
            sr_ratio = kwargs["sr_ratio"]
            ct_size = kwargs.get("ct_size", 1)
            ct_total = ct_size**2 * sr_ratio**2
            ct = jr.normal(key, (ct_total, DIM))
            out, ct_out = model(x, ct, key=key, inference=True)
            assert out.shape == x.shape
            assert ct_out.shape == ct.shape
        elif cls == PartialFormerBlock:
            qa = jr.normal(key, (1, DIM))
            out, qa_out = model(x, qa, key=key, inference=True)
            assert out.shape == x.shape
            assert qa_out.shape == qa.shape
        else:
            out = model(x, key=key, inference=True)
            assert out.shape == x.shape
            
        assert jnp.all(jnp.isfinite(out))

    def test_registry(self):
        assert get_attn("attention") is Attention
        assert get_attn_block("attentionblock") is AttentionBlock
        
    def test_low_precision(self):
        model = Attention(DIM, NUM_HEADS, key=KEY)
        model = jax.tree_util.tree_map(
            lambda leaf: leaf.astype(jnp.bfloat16) if eqx.is_inexact_array(leaf) else leaf,
            model,
        )
        x = jr.normal(KEY, (SEQLEN, DIM)).astype(jnp.bfloat16)
        out = model(x, key=KEY, inference=True)
        assert out.dtype == jnp.bfloat16
        assert jnp.all(jnp.isfinite(out))
