# ty: ignore[invalid-assignment]
# ty: ignore[call-non-callable]
# ty: ignore[too-many-positional-arguments]
__all__ = ["TabPFN", "tabpfn"]

from collections.abc import Callable, Sequence
from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.random as jr
from jaxtyping import Array, Float, Int, PRNGKeyArray

from equimo.core.layers.activation import get_act
from equimo.core.layers.generic import BlockChunk
from equimo.core.layers.norm import get_norm
from equimo.registry import register_model
from equimo.tabular.layers import (
    ColumnAggregator,
    FeatureDistributionEncoder,
    get_attn_block,
    get_decoder,
    get_embedding,
    get_preprocessor,
)
from equimo.utils import make_drop_path_schedule, to_list


@register_model("tabpfn", modality="tabular")
class TabPFN(eqx.Module):
    """TabPFN-style in-context classifier for a single tabular dataset.

    Inputs are unbatched. Use ``jax.vmap`` over datasets if a batch dimension is
    needed.
    """

    preprocessor: eqx.Module
    x_embed: eqx.nn.Linear
    column_label_embedding: eqx.Module
    context_label_embedding: eqx.Module
    feature_encoder: FeatureDistributionEncoder
    column_aggregator: ColumnAggregator
    blocks: Tuple[BlockChunk, ...]
    norm: eqx.Module
    head: eqx.Module

    num_classes: int = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    context_dim: int = eqx.field(static=True)
    depths: Tuple[int, int, int] = eqx.field(static=True)
    num_heads: Tuple[int, int, int] = eqx.field(static=True)
    feature_group_size: int = eqx.field(static=True)
    num_cls_tokens: int = eqx.field(static=True)
    use_nan_indicators: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        num_classes: int = 160,
        dim: int = 128,
        depths: Sequence[int] = (3, 3, 24),
        num_heads: int | Sequence[int] = (8, 8, 8),
        mlp_ratio: float = 2.0,
        feature_group_size: int = 3,
        num_inducing_points: int = 128,
        num_cls_tokens: int = 4,
        num_kv_heads_test: int | None = 1,
        decoder_head_dim: int = 64,
        decoder_num_heads: int = 6,
        decoder_use_softmax_scaling: bool = True,
        use_rope: bool = True,
        rope_base: float = 100_000.0,
        scaling_mlp_hidden_dim: int = 64,
        use_nan_indicators: bool = True,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        context_block: str | type[eqx.Module] = "incontextattentionblock",
        preprocessor_layer: str | type[eqx.Module] = "preprocessor",
        label_embedding_layer: str | type[eqx.Module] = "labelembedding",
        decoder_layer: str | type[eqx.Module] = "attentiondecoder",
        act_layer: str | Callable = "exactgelu",
        norm_layer: str | type[eqx.Module] = "rmsnorm",
        eps: float = 1e-5,
        key: PRNGKeyArray,
    ) -> None:
        if len(depths) != 3:
            raise ValueError("TabPFN expects three depths: feature, column, context.")

        heads = tuple(int(head) for head in to_list(num_heads, 3))
        depths = tuple(int(depth) for depth in depths)
        context_dim = dim * num_cls_tokens
        if dim % heads[0] != 0 or dim % heads[1] != 0:
            raise ValueError("dim must be divisible by feature and column heads.")
        if context_dim % heads[2] != 0:
            raise ValueError("dim * num_cls_tokens must be divisible by context heads.")

        context_block = get_attn_block(context_block)
        preprocessor_layer = get_preprocessor(preprocessor_layer)
        label_embedding_layer = get_embedding(label_embedding_layer)
        decoder_layer = get_decoder(decoder_layer)
        act_layer = get_act(act_layer)
        norm_layer = get_norm(norm_layer)

        key_x, key_col_y, key_ctx_y, key_feature, key_column, key_blocks, key_head = (
            jr.split(key, 7)
        )

        self.preprocessor = preprocessor_layer(
            feature_group_size=feature_group_size,
            use_nan_indicators=use_nan_indicators,
        )
        in_features = feature_group_size * (2 if use_nan_indicators else 1)
        self.x_embed = eqx.nn.Linear(in_features, dim, key=key_x)
        self.column_label_embedding = label_embedding_layer(
            num_classes,
            dim,
            key=key_col_y,
        )
        self.context_label_embedding = label_embedding_layer(
            num_classes,
            context_dim,
            key=key_ctx_y,
        )
        self.feature_encoder = FeatureDistributionEncoder(
            dim,
            heads[0],
            depths[0],
            num_inducing_points=num_inducing_points,
            mlp_ratio=mlp_ratio,
            scaling_mlp_hidden_dim=scaling_mlp_hidden_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
            eps=eps,
            key=key_feature,
        )
        self.column_aggregator = ColumnAggregator(
            dim,
            heads[1],
            depths[1],
            num_cls_tokens=num_cls_tokens,
            mlp_ratio=mlp_ratio,
            use_rope=use_rope,
            rope_base=rope_base,
            act_layer=act_layer,
            norm_layer=norm_layer,
            eps=eps,
            key=key_column,
        )
        dpr = make_drop_path_schedule(
            drop_path_rate,
            [depths[2]],
            uniform=drop_path_uniform,
        )
        self.blocks = (
            BlockChunk(
                depth=depths[2],
                module=context_block,
                module_kwargs={
                    "dim": context_dim,
                    "num_heads": heads[2],
                    "mlp_ratio": mlp_ratio,
                    "num_kv_heads_test": num_kv_heads_test,
                    "scaling_mlp_hidden_dim": scaling_mlp_hidden_dim,
                    "act_layer": act_layer,
                    "norm_layer": norm_layer,
                    "eps": eps,
                },
                drop_path=dpr,
                key=key_blocks,
            ),
        )
        self.norm = norm_layer(context_dim, eps=eps)
        self.head = decoder_layer(
            num_classes,
            context_dim,
            head_dim=decoder_head_dim,
            num_heads=decoder_num_heads,
            use_softmax_scaling=decoder_use_softmax_scaling,
            scaling_mlp_hidden_dim=scaling_mlp_hidden_dim,
            key=key_head,
        )

        self.num_classes = num_classes
        self.dim = dim
        self.context_dim = context_dim
        self.depths = depths
        self.num_heads = heads
        self.feature_group_size = feature_group_size
        self.num_cls_tokens = num_cls_tokens
        self.use_nan_indicators = use_nan_indicators

    def features(
        self,
        x: Float[Array, "rows columns"],
        y: Int[Array, " rows"],
        n_train: int,
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
        **kwargs,
    ) -> Float[Array, "rows dim"]:
        key_feature, key_column, *block_keys = jr.split(key, len(self.blocks) + 2)

        x = self.preprocessor(x, n_train)
        x = jax.vmap(jax.vmap(self.x_embed))(x)

        x = x.at[:n_train].add(self.column_label_embedding(y[:n_train])[:, None, :])
        x = self.feature_encoder(
            x,
            n_train,
            key=key_feature,
            inference=inference,
        )

        cls = self.column_aggregator(x, key=key_column, inference=inference)
        x_context = cls.reshape(cls.shape[0], -1)
        x_context = x_context.at[:n_train].add(
            self.context_label_embedding(y[:n_train])
        )

        for block, block_key in zip(self.blocks, block_keys):
            x_context = block(
                x_context,
                n_train=n_train,
                key=block_key,
                inference=inference,
            )
        return x_context

    def forward_features(
        self,
        x: Float[Array, "rows columns"],
        y: Int[Array, " rows"],
        n_train: int,
        key: PRNGKeyArray,
        inference: Optional[bool] = None,
        **kwargs,
    ) -> dict:
        x = self.features(x, y, n_train, key=key, inference=inference, **kwargs)
        x_norm = jax.vmap(self.norm)(x)
        return {
            "x_norm_train": x_norm[:n_train],
            "x_norm_test": x_norm[n_train:],
            "x_prenorm": x,
        }

    def __call__(
        self,
        x: Float[Array, "rows columns"],
        y: Int[Array, " rows"],
        n_train: int,
        key: PRNGKeyArray = jr.PRNGKey(42),
        inference: Optional[bool] = None,
        **kwargs,
    ) -> Float[Array, "test classes"]:
        x = self.features(x, y, n_train, key=key, inference=inference, **kwargs)
        x = jax.vmap(self.norm)(x)
        return self.head(x[:n_train], x[n_train:], y[:n_train], n_train)


_TABPFN_BASE_CFG: dict = {
    "num_classes": 160,
    "dim": 128,
    "depths": (3, 3, 24),
    "num_heads": (8, 8, 8),
}

_TABPFN_REGISTRY: dict[str, tuple[dict, dict]] = {
    "tabpfn": (_TABPFN_BASE_CFG, {}),
}


def _build_tabpfn(
    variant: str,
    key: PRNGKeyArray | None = None,
    **overrides,
) -> TabPFN:
    if key is None:
        key = jax.random.PRNGKey(42)

    base_cfg, variant_cfg = _TABPFN_REGISTRY[variant]
    cfg = base_cfg | variant_cfg | overrides
    return TabPFN(**cfg, key=key)


def tabpfn(
    key: PRNGKeyArray | None = None,
    **kwargs,
) -> TabPFN:
    """Build a TabPFN model with default Equimo tabular settings."""
    return _build_tabpfn("tabpfn", key=key, **kwargs)
