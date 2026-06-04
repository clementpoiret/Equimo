"""TabPFN v3 (Prior Labs) in JAX/Equinox.

Single-dataset (unbatched) modules — vmap over the batch axis if needed. Module
attribute names mirror the PyTorch reference (``tabpfn_v3.py``) so the pretrained
state dict maps on with near-zero renaming (see ``loader.py``).

Shape suffixes: R rows (first N train, rest M test), C columns, E embed dim,
D ICL dim (= Cl*E), Cl CLS tokens, H heads, Dh head dim, T classes.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Int, PRNGKeyArray

from equimo.core.layers.activation import get_act

gelu = get_act("exactgelu")  # torch nn.GELU() is the erf-exact variant
# torch nn.RMSNorm(elementwise_affine=True) has weight only and eps=None,
# which resolves to finfo(float32).eps. Equinox defaults differ on both.
_RMS_EPS = float(jnp.finfo(jnp.float32).eps)


def _rms(dim: int) -> eqx.nn.RMSNorm:
    return eqx.nn.RMSNorm(dim, eps=_RMS_EPS, use_bias=False)


def _make_mlp(emsize: int, dim_ff: int, *, key: PRNGKeyArray) -> list:
    """Bias-free 2-layer GELU MLP as a ``[Linear, gelu, Linear]`` list so its
    pytree paths are ``mlp.0`` / ``mlp.2`` (matching ``nn.Sequential``)."""
    k1, k2 = jr.split(key)
    return [
        eqx.nn.Linear(emsize, dim_ff, use_bias=False, key=k1),
        gelu,
        eqx.nn.Linear(dim_ff, emsize, use_bias=False, key=k2),
    ]


def _ffn_vec(mlp: list, v: Array) -> Array:
    l1, act, l2 = mlp
    return l2(act(l1(v)))


def _ffn(mlp: list, x_SE: Array) -> Array:
    return jax.vmap(lambda t: _ffn_vec(mlp, t))(x_SE)


def _to_heads(proj: eqx.nn.Linear, x_SE: Array, num_heads: int, head_dim: int) -> Array:
    """(S, E) -> (H, S, Dh)."""
    out = jax.vmap(proj)(x_SE)
    return out.reshape(x_SE.shape[0], num_heads, head_dim).transpose(1, 0, 2)


def _sdpa(
    q_HSd: Array,
    k_HSd: Array,
    v_HSd: Array,
    head_dim: int,
    scaling: eqx.Module | None = None,
    n: Array | int | None = None,
) -> Array:
    """Scaled dot-product attention with optional query scaling. -> (H, Sq, Dh)."""
    if scaling is not None:
        q_HSd = scaling(q_HSd, n)
    if k_HSd.shape[0] != q_HSd.shape[0]:  # GQA: broadcast kv heads over query heads
        k_HSd = jnp.repeat(k_HSd, q_HSd.shape[0] // k_HSd.shape[0], axis=0)
        v_HSd = jnp.repeat(v_HSd, q_HSd.shape[0] // v_HSd.shape[0], axis=0)
    scores = jnp.einsum("hqd,hkd->hqk", q_HSd, k_HSd) / jnp.sqrt(head_dim)
    attn = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("hqk,hkd->hqd", attn, v_HSd)


class SoftmaxScalingMLP(eqx.Module):
    """Query scaling: ``q * base_mlp(log n) * (1 + tanh(query_mlp(q)))``."""

    base_mlp: list
    query_mlp: list
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(
        self, num_heads: int, head_dim: int, n_hidden: int = 64, *, key: PRNGKeyArray
    ) -> None:
        k1, k2, k3, k4 = jr.split(key, 4)
        self.base_mlp = [
            eqx.nn.Linear(1, n_hidden, key=k1),
            gelu,
            eqx.nn.Linear(n_hidden, num_heads * head_dim, key=k2),
        ]
        self.query_mlp = [
            eqx.nn.Linear(head_dim, n_hidden, key=k3),
            gelu,
            eqx.nn.Linear(n_hidden, head_dim, key=k4),
        ]
        self.num_heads = num_heads
        self.head_dim = head_dim

    def __call__(self, q_HSd: Array, n: Array | int) -> Array:
        logn = jnp.log(jnp.maximum(jnp.asarray(n, q_HSd.dtype), 1.0)).reshape(1)
        base = _ffn_vec(self.base_mlp, logn).reshape(self.num_heads, 1, self.head_dim)
        modulation = 1 + jnp.tanh(
            jax.vmap(jax.vmap(lambda t: _ffn_vec(self.query_mlp, t)))(q_HSd)
        )
        return q_HSd * base * modulation


class Attention(eqx.Module):
    """Bias-free multi-head self/cross attention, optional RoPE."""

    q_projection: eqx.nn.Linear
    k_projection: eqx.nn.Linear
    v_projection: eqx.nn.Linear
    out_projection: eqx.nn.Linear
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(
        self, embedding_size: int, num_heads: int, head_dim: int, *, key: PRNGKeyArray
    ) -> None:
        k1, k2, k3, k4 = jr.split(key, 4)
        inner = num_heads * head_dim
        self.q_projection = eqx.nn.Linear(embedding_size, inner, use_bias=False, key=k1)
        self.k_projection = eqx.nn.Linear(embedding_size, inner, use_bias=False, key=k2)
        self.v_projection = eqx.nn.Linear(embedding_size, inner, use_bias=False, key=k3)
        self.out_projection = eqx.nn.Linear(
            inner, embedding_size, use_bias=False, key=k4
        )
        self.num_heads = num_heads
        self.head_dim = head_dim

    def cross(
        self, x_q_QE: Array, x_kv_VE: Array, rope: eqx.Module | None = None
    ) -> Array:
        q = _to_heads(self.q_projection, x_q_QE, self.num_heads, self.head_dim)
        k = _to_heads(self.k_projection, x_kv_VE, self.num_heads, self.head_dim)
        v = _to_heads(self.v_projection, x_kv_VE, self.num_heads, self.head_dim)
        if rope is not None:
            q = jax.vmap(rope)(q)
            k = jax.vmap(rope)(k)
        out = _sdpa(q, k, v, self.head_dim)
        out = out.transpose(1, 0, 2).reshape(x_q_QE.shape[0], -1)
        return jax.vmap(self.out_projection)(out)

    def __call__(self, x_SE: Array, rope: eqx.Module | None = None) -> Array:
        return self.cross(x_SE, x_SE, rope)


class CrossAttention(eqx.Module):
    """Query attends to key/value, with optional softmax scaling on queries."""

    q_projection: eqx.nn.Linear
    k_projection: eqx.nn.Linear
    v_projection: eqx.nn.Linear
    out_projection: eqx.nn.Linear
    softmax_scaling_layer: SoftmaxScalingMLP | None
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        head_dim: int,
        softmax_scaling_layer: SoftmaxScalingMLP | None = None,
        *,
        key: PRNGKeyArray,
    ) -> None:
        k1, k2, k3, k4 = jr.split(key, 4)
        inner = num_heads * head_dim
        self.q_projection = eqx.nn.Linear(embedding_size, inner, use_bias=False, key=k1)
        self.k_projection = eqx.nn.Linear(embedding_size, inner, use_bias=False, key=k2)
        self.v_projection = eqx.nn.Linear(embedding_size, inner, use_bias=False, key=k3)
        self.out_projection = eqx.nn.Linear(
            inner, embedding_size, use_bias=False, key=k4
        )
        self.softmax_scaling_layer = softmax_scaling_layer
        self.num_heads = num_heads
        self.head_dim = head_dim

    def __call__(self, x_q_QE: Array, x_kv_VE: Array) -> Array:
        q = _to_heads(self.q_projection, x_q_QE, self.num_heads, self.head_dim)
        k = _to_heads(self.k_projection, x_kv_VE, self.num_heads, self.head_dim)
        v = _to_heads(self.v_projection, x_kv_VE, self.num_heads, self.head_dim)
        out = _sdpa(
            q, k, v, self.head_dim, self.softmax_scaling_layer, x_kv_VE.shape[0]
        )
        out = out.transpose(1, 0, 2).reshape(x_q_QE.shape[0], -1)
        return jax.vmap(self.out_projection)(out)


class ICLAttention(eqx.Module):
    """Self-attention with keys/values restricted to the first ``n_train`` rows.

    Train rows attend with all heads; test rows attend with only the first
    ``num_kv_heads_test`` KV heads (MQA/GQA), matching the reference inference.
    """

    q_projection: eqx.nn.Linear
    k_projection: eqx.nn.Linear
    v_projection: eqx.nn.Linear
    out_projection: eqx.nn.Linear
    softmax_scaling_layer: SoftmaxScalingMLP | None
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    num_kv_heads_test: int | None = eqx.field(static=True)

    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        head_dim: int,
        softmax_scaling_layer: SoftmaxScalingMLP | None = None,
        num_kv_heads_test: int | None = None,
        *,
        key: PRNGKeyArray,
    ) -> None:
        k1, k2, k3, k4 = jr.split(key, 4)
        inner = num_heads * head_dim
        self.q_projection = eqx.nn.Linear(embedding_size, inner, use_bias=False, key=k1)
        self.k_projection = eqx.nn.Linear(embedding_size, inner, use_bias=False, key=k2)
        self.v_projection = eqx.nn.Linear(embedding_size, inner, use_bias=False, key=k3)
        self.out_projection = eqx.nn.Linear(
            inner, embedding_size, use_bias=False, key=k4
        )
        self.softmax_scaling_layer = softmax_scaling_layer
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads_test = num_kv_heads_test

    def __call__(self, x_RD: Array, n_train: int) -> Array:
        scl = self.softmax_scaling_layer
        q = _to_heads(self.q_projection, x_RD, self.num_heads, self.head_dim)
        k = _to_heads(self.k_projection, x_RD, self.num_heads, self.head_dim)[
            :, :n_train
        ]
        v = _to_heads(self.v_projection, x_RD, self.num_heads, self.head_dim)[
            :, :n_train
        ]

        if self.num_kv_heads_test is None:
            out = _sdpa(q, k, v, self.head_dim, scl, n_train)
        else:
            kt = self.num_kv_heads_test
            out_train = _sdpa(q[:, :n_train], k, v, self.head_dim, scl, n_train)
            out_test = _sdpa(
                q[:, n_train:], k[:kt], v[:kt], self.head_dim, scl, n_train
            )
            out = jnp.concatenate([out_train, out_test], axis=1)

        out = out.transpose(1, 0, 2).reshape(x_RD.shape[0], -1)
        return jax.vmap(self.out_projection)(out)


class CrossAttentionBlock(eqx.Module):
    """Pre-norm cross-attention + MLP."""

    attn: CrossAttention
    mlp: list
    layernorm_q: eqx.nn.RMSNorm
    layernorm_kv: eqx.nn.RMSNorm
    layernorm2: eqx.nn.RMSNorm

    def __init__(
        self,
        emsize: int,
        nhead: int,
        dim_feedforward: int,
        softmax_scaling_layer: SoftmaxScalingMLP | None = None,
        *,
        key: PRNGKeyArray,
    ) -> None:
        k1, k2 = jr.split(key)
        self.attn = CrossAttention(
            emsize, nhead, emsize // nhead, softmax_scaling_layer, key=k1
        )
        self.mlp = _make_mlp(emsize, dim_feedforward, key=k2)
        self.layernorm_q = _rms(emsize)
        self.layernorm_kv = _rms(emsize)
        self.layernorm2 = _rms(emsize)

    def __call__(self, x_q_QE: Array, context_VE: Array) -> Array:
        x = x_q_QE + self.attn(
            jax.vmap(self.layernorm_q)(x_q_QE), jax.vmap(self.layernorm_kv)(context_VE)
        )
        return x + _ffn(self.mlp, jax.vmap(self.layernorm2)(x))


class TransformerBlock(eqx.Module):
    """Pre-norm self-attention + MLP, with a CLS-readout cross variant."""

    attention: Attention
    mlp: list
    layernorm: eqx.nn.RMSNorm
    layernorm_mlp: eqx.nn.RMSNorm

    def __init__(
        self, emsize: int, nhead: int, dim_feedforward: int, *, key: PRNGKeyArray
    ) -> None:
        k1, k2 = jr.split(key)
        self.attention = Attention(emsize, nhead, emsize // nhead, key=k1)
        self.mlp = _make_mlp(emsize, dim_feedforward, key=k2)
        self.layernorm = _rms(emsize)
        self.layernorm_mlp = _rms(emsize)

    def __call__(self, x_SE: Array, rope: eqx.Module | None = None) -> Array:
        x = x_SE + self.attention(jax.vmap(self.layernorm)(x_SE), rope)
        return x + _ffn(self.mlp, jax.vmap(self.layernorm_mlp)(x))

    def forward_cross(
        self, q_QE: Array, kv_VE: Array, rope: eqx.Module | None = None
    ) -> Array:
        x = q_QE + self.attention.cross(
            jax.vmap(self.layernorm)(q_QE), jax.vmap(self.layernorm)(kv_VE), rope
        )
        return x + _ffn(self.mlp, jax.vmap(self.layernorm_mlp)(x))


class ICLTransformerBlock(eqx.Module):
    """Pre-norm ICL block: keys/values restricted to train rows."""

    icl_attention: ICLAttention
    mlp: list
    layernorm: eqx.nn.RMSNorm
    layernorm_mlp: eqx.nn.RMSNorm

    def __init__(
        self,
        emsize: int,
        nhead: int,
        dim_feedforward: int,
        softmax_scaling_layer: SoftmaxScalingMLP | None = None,
        num_kv_heads_test: int | None = None,
        *,
        key: PRNGKeyArray,
    ) -> None:
        k1, k2 = jr.split(key)
        self.icl_attention = ICLAttention(
            emsize,
            nhead,
            emsize // nhead,
            softmax_scaling_layer,
            num_kv_heads_test,
            key=k1,
        )
        self.mlp = _make_mlp(emsize, dim_feedforward, key=k2)
        self.layernorm = _rms(emsize)
        self.layernorm_mlp = _rms(emsize)

    def __call__(self, x_RD: Array, n_train: int) -> Array:
        x = x_RD + self.icl_attention(jax.vmap(self.layernorm)(x_RD), n_train)
        return x + _ffn(self.mlp, jax.vmap(self.layernorm_mlp)(x))


class InducedSelfAttentionBlock(eqx.Module):
    """SetTransformer-style induced attention over rows (per column)."""

    cross_attn_block1: CrossAttentionBlock
    cross_attn_block2: CrossAttentionBlock
    inducing_vectors: Array

    def __init__(
        self,
        emsize: int,
        nhead: int,
        num_inducing_points: int,
        dim_feedforward: int,
        softmax_scaling_layer: SoftmaxScalingMLP | None = None,
        *,
        key: PRNGKeyArray,
    ) -> None:
        k1, k2, k3 = jr.split(key, 3)
        self.cross_attn_block1 = CrossAttentionBlock(
            emsize, nhead, dim_feedforward, softmax_scaling_layer, key=k1
        )
        self.cross_attn_block2 = CrossAttentionBlock(
            emsize, nhead, dim_feedforward, key=k2
        )
        self.inducing_vectors = (
            jr.truncated_normal(k3, -2, 2, (num_inducing_points, emsize)) * 0.02
        )

    def __call__(self, x_RE: Array, n_train: int) -> Array:
        hidden = self.cross_attn_block1(self.inducing_vectors, x_RE[:n_train])
        return self.cross_attn_block2(x_RE, hidden)


class FeatureDistributionEmbedder(eqx.Module):
    """Stack of induced self-attention blocks applied independently per column."""

    layers: list

    def __init__(
        self,
        emsize: int,
        nhead: int,
        num_inducing_points: int,
        dim_feedforward: int,
        num_layers: int,
        softmax_scaling_mlp_hidden_dim: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        keys = jr.split(key, num_layers)
        self.layers = []
        for k in keys:
            ks, kb = jr.split(k)
            scaling = SoftmaxScalingMLP(
                nhead, emsize // nhead, softmax_scaling_mlp_hidden_dim, key=ks
            )
            self.layers.append(
                InducedSelfAttentionBlock(
                    emsize, nhead, num_inducing_points, dim_feedforward, scaling, key=kb
                )
            )

    def __call__(self, x_RCE: Array, n_train: int) -> Array:
        x_CRE = x_RCE.transpose(1, 0, 2)
        for layer in self.layers:
            x_CRE = jax.vmap(layer, in_axes=(0, None))(x_CRE, n_train)
        return x_CRE.transpose(1, 0, 2)


class ColumnAggregator(eqx.Module):
    """Per-row cross-feature interaction: CLS tokens aggregate column info."""

    blocks: list
    cls_tokens: Array
    rope: eqx.nn.RotaryPositionalEmbedding | None
    out_ln: eqx.nn.RMSNorm
    num_cls_tokens: int = eqx.field(static=True)

    def __init__(
        self,
        emsize: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        num_cls_tokens: int,
        use_rope: bool = True,
        rope_base: float = 100_000.0,
        *,
        key: PRNGKeyArray,
    ) -> None:
        keys = jr.split(key, num_layers + 1)
        self.blocks = [
            TransformerBlock(emsize, nhead, dim_feedforward, key=k)
            for k in keys[:num_layers]
        ]
        self.cls_tokens = (
            jr.truncated_normal(keys[-1], -2, 2, (num_cls_tokens, emsize)) * 0.02
        )
        self.rope = (
            eqx.nn.RotaryPositionalEmbedding(emsize // nhead, theta=rope_base)
            if use_rope
            else None
        )
        self.out_ln = _rms(emsize)
        self.num_cls_tokens = num_cls_tokens

    def _row(self, x_CE: Array) -> Array:
        x = jnp.concatenate([self.cls_tokens, x_CE], axis=0)
        for block in self.blocks[:-1]:
            x = block(x, self.rope)
        cls_out = self.blocks[-1].forward_cross(x[: self.num_cls_tokens], x, self.rope)
        return jax.vmap(self.out_ln)(cls_out)

    def __call__(self, x_RCE: Array) -> Array:
        return jax.vmap(self._row)(x_RCE)


class TrainableOrthogonalEmbedding(eqx.Module):
    """Class-label embedding (weights loaded; init is irrelevant)."""

    embedding: eqx.nn.Embedding

    def __init__(self, num_classes: int, embed_dim: int, *, key: PRNGKeyArray) -> None:
        self.embedding = eqx.nn.Embedding(num_classes, embed_dim, key=key)

    def __call__(self, idx_R: Int[Array, " R"]) -> Float[Array, "R E"]:
        return jax.vmap(self.embedding)(idx_R)


class ManyClassDecoder(eqx.Module):
    """Attention retrieval decoder: test rows read a weighted average of one-hot
    train targets, then take the log to get logits."""

    q_projection: eqx.nn.Linear
    k_projection: eqx.nn.Linear
    softmax_scaling_layer: SoftmaxScalingMLP | None
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    max_num_classes: int = eqx.field(static=True)

    def __init__(
        self,
        max_num_classes: int,
        input_size: int,
        head_dim: int = 64,
        num_heads: int = 6,
        softmax_scaling_layer: SoftmaxScalingMLP | None = None,
        *,
        key: PRNGKeyArray,
    ) -> None:
        k1, k2 = jr.split(key)
        inner = head_dim * num_heads
        self.q_projection = eqx.nn.Linear(input_size, inner, key=k1)
        self.k_projection = eqx.nn.Linear(input_size, inner, key=k2)
        self.softmax_scaling_layer = softmax_scaling_layer
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_num_classes = max_num_classes

    def __call__(
        self,
        train_embeddings: Float[Array, "N D"],
        test_embeddings: Float[Array, "M D"],
        targets: Int[Array, " N"],
        n_train: int,
    ) -> Float[Array, "M T"]:
        q = _to_heads(self.q_projection, test_embeddings, self.num_heads, self.head_dim)
        k = _to_heads(
            self.k_projection, train_embeddings, self.num_heads, self.head_dim
        )
        one_hot = jax.nn.one_hot(targets, self.max_num_classes)
        if self.softmax_scaling_layer is not None:
            q = self.softmax_scaling_layer(q, n_train)
        scores = jnp.einsum("hmd,hnd->hmn", q, k) / jnp.sqrt(self.head_dim)
        attn = jax.nn.softmax(scores, axis=-1)
        probs = jnp.einsum("hmn,nt->hmt", attn, one_hot).mean(0)
        return jnp.log(jnp.clip(probs, min=1e-5) + 3e-5)


_NAN_IND, _POSINF_IND, _NEGINF_IND = -2.0, 2.0, 4.0


class TabPFNV3(eqx.Module):
    """TabPFN v3 forward pass for a single dataset."""

    x_embed: eqx.nn.Linear
    col_y_encoder: TrainableOrthogonalEmbedding
    icl_y_encoder: TrainableOrthogonalEmbedding
    feature_distribution_embedder: FeatureDistributionEmbedder
    column_aggregator: ColumnAggregator
    icl_blocks: list
    output_norm: eqx.nn.RMSNorm
    many_class_decoder: ManyClassDecoder
    max_num_classes: int = eqx.field(static=True)
    feature_group_size: int = eqx.field(static=True)
    use_nan_indicators: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        max_num_classes: int = 160,
        embed_dim: int = 128,
        dist_embed_num_blocks: int = 3,
        dist_embed_num_heads: int = 8,
        dist_embed_num_inducing_points: int = 128,
        feature_group_size: int = 3,
        feat_agg_num_blocks: int = 3,
        feat_agg_num_heads: int = 8,
        feat_agg_num_cls_tokens: int = 4,
        feat_agg_rope_base: float = 100_000.0,
        use_rope: bool = True,
        nlayers: int = 24,
        icl_num_heads: int = 8,
        icl_num_kv_heads_test: int | None = 1,
        decoder_head_dim: int = 64,
        decoder_num_heads: int = 6,
        decoder_use_softmax_scaling: bool = True,
        ff_factor: int = 2,
        softmax_scaling_mlp_hidden_dim: int = 64,
        use_nan_indicators: bool = True,
        key: PRNGKeyArray,
    ) -> None:
        ks = jr.split(key, 7)
        icl_emsize = embed_dim * feat_agg_num_cls_tokens
        in_features = feature_group_size * (2 if use_nan_indicators else 1)

        self.x_embed = eqx.nn.Linear(in_features, embed_dim, key=ks[0])
        self.col_y_encoder = TrainableOrthogonalEmbedding(
            max_num_classes, embed_dim, key=ks[1]
        )
        self.icl_y_encoder = TrainableOrthogonalEmbedding(
            max_num_classes, icl_emsize, key=ks[2]
        )
        self.feature_distribution_embedder = FeatureDistributionEmbedder(
            embed_dim,
            dist_embed_num_heads,
            dist_embed_num_inducing_points,
            embed_dim * ff_factor,
            dist_embed_num_blocks,
            softmax_scaling_mlp_hidden_dim,
            key=ks[3],
        )
        self.column_aggregator = ColumnAggregator(
            embed_dim,
            feat_agg_num_heads,
            feat_agg_num_blocks,
            embed_dim * ff_factor,
            feat_agg_num_cls_tokens,
            use_rope,
            feat_agg_rope_base,
            key=ks[4],
        )
        icl_keys = jr.split(ks[5], nlayers)
        self.icl_blocks = []
        for k in icl_keys:
            ksc, kb = jr.split(k)
            scaling = SoftmaxScalingMLP(
                icl_num_heads,
                icl_emsize // icl_num_heads,
                softmax_scaling_mlp_hidden_dim,
                key=ksc,
            )
            self.icl_blocks.append(
                ICLTransformerBlock(
                    icl_emsize,
                    icl_num_heads,
                    icl_emsize * ff_factor,
                    scaling,
                    icl_num_kv_heads_test,
                    key=kb,
                )
            )
        self.output_norm = _rms(icl_emsize)
        decoder_scaling = (
            SoftmaxScalingMLP(
                decoder_num_heads,
                decoder_head_dim,
                softmax_scaling_mlp_hidden_dim,
                key=jr.split(ks[6])[0],
            )
            if decoder_use_softmax_scaling
            else None
        )
        self.many_class_decoder = ManyClassDecoder(
            max_num_classes,
            icl_emsize,
            decoder_head_dim,
            decoder_num_heads,
            decoder_scaling,
            key=ks[6],
        )
        self.max_num_classes = max_num_classes
        self.feature_group_size = feature_group_size
        self.use_nan_indicators = use_nan_indicators

    def _preprocess(self, x_RC: Float[Array, "R C"], n_train: int) -> Array:
        """NaN/Inf indicators -> impute -> standardize -> circular-shift groups."""
        finite = jnp.isfinite(x_RC)
        ind = None
        if self.use_nan_indicators:
            ind = (
                jnp.isnan(x_RC) * _NAN_IND
                + jnp.isposinf(x_RC) * _POSINF_IND
                + jnp.isneginf(x_RC) * _NEGINF_IND
            )
        x_train = jnp.where(finite[:n_train], x_RC[:n_train], jnp.nan)
        means = jnp.nan_to_num(jnp.nanmean(x_train, axis=0), nan=0.0)
        x = jnp.where(finite, x_RC, means[None, :])

        xt = x[:n_train]
        mean = xt.mean(axis=0, keepdims=True)
        std = xt.std(axis=0, ddof=1, keepdims=True) + 1e-20
        x = jnp.clip((x - mean) / std, min=-100, max=100)

        g = self.feature_group_size
        groups = [jnp.roll(x, -(2**i), axis=1) for i in range(g)]
        out = jnp.stack(groups, axis=-1)
        if ind is not None:
            ind_groups = [jnp.roll(ind, -(2**i), axis=1) for i in range(g)]
            out = jnp.concatenate([out, jnp.stack(ind_groups, axis=-1)], axis=-1)
        return out

    def __call__(
        self,
        x_RC: Float[Array, "R C"],
        y_R: Float[Array, " R"],
        n_train: int,
    ) -> Float[Array, "M T"]:
        idx = jnp.clip(y_R.astype(jnp.int32), min=0, max=self.max_num_classes - 1)

        x = self._preprocess(x_RC, n_train)
        x_RCE = jax.vmap(jax.vmap(self.x_embed))(x)

        y_col = self.col_y_encoder(idx)
        x_RCE = x_RCE.at[:n_train].add(y_col[:n_train, None, :])

        x_RCE = self.feature_distribution_embedder(x_RCE, n_train)
        cls = self.column_aggregator(x_RCE)
        x_RD = cls.reshape(cls.shape[0], -1)

        x_RD = x_RD.at[:n_train].add(self.icl_y_encoder(idx)[:n_train])
        for block in self.icl_blocks:
            x_RD = block(x_RD, n_train)
        x_RD = jax.vmap(self.output_norm)(x_RD)

        return self.many_class_decoder(
            x_RD[:n_train], x_RD[n_train:], idx[:n_train], n_train
        )


@eqx.filter_jit
def predict(
    model: TabPFNV3,
    x: Float[Array, "R C"],
    y: Float[Array, " R"],
    n_train: int,
) -> Float[Array, "M T"]:
    return model(x, y, n_train)
