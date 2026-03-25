"""Tests for equimo.layers.ffn."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from equimo.layers.ffn import (
    DINOHead,
    Mlp,
    SwiGlu,
    SwiGluFused,
    WeightNormLinear,
    get_ffn,
)

# Shared fixtures

KEY = jr.PRNGKey(0)
SEQLEN = 16
DIM = 64


# WeightNormLinear


class TestWeightNormLinear:
    def test_output_shape(self):
        layer = WeightNormLinear(DIM, 32, key=KEY)
        x = jnp.ones((SEQLEN, DIM))
        assert layer(x).shape == (SEQLEN, 32)

    def test_output_shape_square(self):
        layer = WeightNormLinear(DIM, DIM, key=KEY)
        x = jnp.ones((SEQLEN, DIM))
        assert layer(x).shape == (SEQLEN, DIM)

    def test_output_finite(self):
        layer = WeightNormLinear(DIM, 32, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert jnp.all(jnp.isfinite(layer(x)))

    def test_weight_normalization_property(self):
        """Each row of the effective weight matrix should have unit L2 norm."""
        layer = WeightNormLinear(DIM, 32, key=KEY)
        v_norm = jnp.linalg.norm(layer.weight_v, axis=1, keepdims=True)
        normalized = layer.weight_v / v_norm
        row_norms = jnp.linalg.norm(normalized, axis=1)
        assert jnp.allclose(row_norms, jnp.ones_like(row_norms), atol=1e-6)

    def test_weight_g_initial_shape(self):
        out_features = 32
        layer = WeightNormLinear(DIM, out_features, key=KEY)
        assert layer.weight_g.shape == (out_features, 1)

    def test_weight_g_initial_ones(self):
        layer = WeightNormLinear(DIM, 32, key=KEY)
        assert jnp.all(layer.weight_g == 1.0)


# DINOHead


class TestDINOHead:
    def test_output_shape(self):
        head = DINOHead(DIM, 32, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert head(x).shape == (SEQLEN, 32)

    def test_output_finite(self):
        head = DINOHead(DIM, 32, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert jnp.all(jnp.isfinite(head(x)))

    def test_key_and_inference_optional(self):
        """key and inference args are unused; both call forms must produce identical output."""
        head = DINOHead(DIM, 32, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        out_bare = head(x)
        out_with_args = head(x, inference=True, key=KEY)
        assert jnp.allclose(out_bare, out_with_args)

    def test_deterministic_across_keys(self):
        """DINOHead has no stochastic ops; output must not depend on key."""
        head = DINOHead(DIM, 32, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        out1 = head(x, key=jr.PRNGKey(1))
        out2 = head(x, key=jr.PRNGKey(2))
        assert jnp.allclose(out1, out2)

    def test_bfloat16_input(self):
        """bfloat16 inputs must not produce NaN/Inf (eps selection check)."""
        head = DINOHead(DIM, 32, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM)).astype(jnp.bfloat16)
        out = head(x)
        assert jnp.all(jnp.isfinite(out))

    def test_float16_input(self):
        """float16 inputs must not produce NaN/Inf (eps selection check)."""
        head = DINOHead(DIM, 32, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM)).astype(jnp.float16)
        out = head(x)
        assert jnp.all(jnp.isfinite(out))

    def test_custom_hidden_and_bottleneck(self):
        head = DINOHead(DIM, 16, hidden_features=128, bottleneck_features=32, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert head(x).shape == (SEQLEN, 16)

    def test_l2_norm_before_last_layer(self):
        """Intermediate features after fc3+act are L2-normalised row-wise."""
        head = DINOHead(DIM, 32, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))

        # Replicate the forward pass up to the norm step
        act = head.act_layer
        h = act(jax.vmap(head.fc1)(x))
        h = act(jax.vmap(head.fc2)(h))
        h = act(jax.vmap(head.fc3)(h))
        norms = jnp.linalg.norm(h, axis=-1)
        assert jnp.all(norms > 0)  # sanity: nonzero before normalisation

        eps = 1e-12
        h_normed = h / (norms[..., None] + eps)
        row_norms = jnp.linalg.norm(h_normed, axis=-1)
        assert jnp.allclose(row_norms, jnp.ones_like(row_norms), atol=1e-5)


# Mlp


class TestMlp:
    def test_output_shape_default(self):
        mlp = Mlp(DIM, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert mlp(x, KEY).shape == (SEQLEN, DIM)

    def test_output_shape_custom_out(self):
        mlp = Mlp(DIM, out_dim=32, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert mlp(x, KEY).shape == (SEQLEN, 32)

    def test_output_shape_custom_hidden(self):
        mlp = Mlp(DIM, hidden_dim=256, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert mlp(x, KEY).shape == (SEQLEN, DIM)

    def test_output_finite_train(self):
        mlp = Mlp(DIM, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert jnp.all(jnp.isfinite(mlp(x, KEY, inference=False)))

    def test_output_finite_inference(self):
        mlp = Mlp(DIM, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert jnp.all(jnp.isfinite(mlp(x, KEY, inference=True)))

    def test_deterministic_in_inference_mode(self):
        mlp = Mlp(DIM, dropout_rate=0.5, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        out1 = mlp(x, jr.PRNGKey(1), inference=True)
        out2 = mlp(x, jr.PRNGKey(2), inference=True)
        assert jnp.allclose(out1, out2)

    def test_stochastic_in_train_mode(self):
        mlp = Mlp(DIM, dropout_rate=0.9, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        out1 = mlp(x, jr.PRNGKey(1), inference=False)
        out2 = mlp(x, jr.PRNGKey(2), inference=False)
        assert not jnp.allclose(out1, out2)

    def test_zero_dropout_train_equals_inference(self):
        mlp = Mlp(DIM, dropout_rate=0.0, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        out_train = mlp(x, KEY, inference=False)
        out_infer = mlp(x, KEY, inference=True)
        assert jnp.allclose(out_train, out_infer)

    def test_mask_zeros_tokens(self):
        """A mask of zeros should produce an all-zero output."""
        mlp = Mlp(DIM, dropout_rate=0.0, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        mask = jnp.zeros((SEQLEN, 1))
        out = mlp(x, KEY, mask=mask, inference=True)
        assert jnp.allclose(out, jnp.zeros_like(out))

    def test_mask_ones_is_identity(self):
        """A mask of ones must leave the output unchanged."""
        mlp = Mlp(DIM, dropout_rate=0.0, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        out_no_mask = mlp(x, KEY, inference=True)
        out_ones_mask = mlp(x, KEY, mask=jnp.ones((SEQLEN, 1)), inference=True)
        assert jnp.allclose(out_no_mask, out_ones_mask)

    def test_norm_layer(self):
        import equinox as eqx

        mlp = Mlp(DIM, norm_layer=eqx.nn.LayerNorm, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        out = mlp(x, KEY, inference=True)
        assert out.shape == (SEQLEN, DIM)
        assert jnp.all(jnp.isfinite(out))

    def test_no_bias(self):
        mlp = Mlp(DIM, bias=False, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert jnp.all(jnp.isfinite(mlp(x, KEY, inference=True)))


# SwiGlu


class TestSwiGlu:
    def test_output_shape_default(self):
        sg = SwiGlu(DIM, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert sg(x, KEY).shape == (SEQLEN, DIM)

    def test_output_shape_custom_out(self):
        sg = SwiGlu(DIM, out_dim=32, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert sg(x, KEY).shape == (SEQLEN, 32)

    def test_output_finite(self):
        sg = SwiGlu(DIM, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert jnp.all(jnp.isfinite(sg(x, KEY)))

    def test_hidden_dim_aligned(self):
        """Internal hidden dimension must be a multiple of align_to."""
        align_to = 8
        sg = SwiGlu(DIM, hidden_dim=100, align_to=align_to, key=KEY)
        hidden = sg.w1.out_features
        assert hidden % align_to == 0

    def test_custom_align_to(self):
        sg = SwiGlu(DIM, hidden_dim=100, align_to=16, key=KEY)
        assert sg.w1.out_features % 16 == 0

    def test_w1_w2_same_hidden_dim(self):
        sg = SwiGlu(DIM, hidden_dim=100, key=KEY)
        assert sg.w1.out_features == sg.w2.out_features

    def test_deterministic_in_inference_mode(self):
        sg = SwiGlu(DIM, dropout_rate=0.5, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        out1 = sg(x, jr.PRNGKey(1), inference=True)
        out2 = sg(x, jr.PRNGKey(2), inference=True)
        assert jnp.allclose(out1, out2)

    def test_stochastic_in_train_mode(self):
        sg = SwiGlu(DIM, dropout_rate=0.9, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        out1 = sg(x, jr.PRNGKey(1), inference=False)
        out2 = sg(x, jr.PRNGKey(2), inference=False)
        assert not jnp.allclose(out1, out2)

    def test_no_bias(self):
        sg = SwiGlu(DIM, bias=False, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert jnp.all(jnp.isfinite(sg(x, KEY, inference=True)))


# SwiGluFused


class TestSwiGluFused:
    def test_output_shape_default(self):
        sgf = SwiGluFused(DIM, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert sgf(x, KEY).shape == (SEQLEN, DIM)

    def test_output_shape_custom_out(self):
        sgf = SwiGluFused(DIM, out_dim=32, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert sgf(x, KEY).shape == (SEQLEN, 32)

    def test_output_finite(self):
        sgf = SwiGluFused(DIM, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert jnp.all(jnp.isfinite(sgf(x, KEY)))

    def test_hidden_dim_aligned(self):
        """Internal hidden dimension must be a multiple of align_to."""
        align_to = 8
        sgf = SwiGluFused(DIM, hidden_dim=100, align_to=align_to, key=KEY)
        # w12 projects to 2 * hidden, so hidden = w12.out_features // 2
        hidden = sgf.w12.out_features // 2
        assert hidden % align_to == 0

    def test_custom_align_to(self):
        sgf = SwiGluFused(DIM, hidden_dim=100, align_to=16, key=KEY)
        hidden = sgf.w12.out_features // 2
        assert hidden % 16 == 0

    def test_fused_hidden_matches_unfused(self):
        """SwiGluFused and SwiGlu must produce the same hidden dim for equal args."""
        hidden_dim = 100
        sg = SwiGlu(DIM, hidden_dim=hidden_dim, key=KEY)
        sgf = SwiGluFused(DIM, hidden_dim=hidden_dim, key=KEY)
        assert sg.w1.out_features == sgf.w12.out_features // 2

    def test_w12_is_double_hidden(self):
        """w12 output dim must be exactly 2 * hidden."""
        sgf = SwiGluFused(DIM, hidden_dim=128, key=KEY)
        hidden = sgf.w3.in_features
        assert sgf.w12.out_features == 2 * hidden

    def test_deterministic_in_inference_mode(self):
        sgf = SwiGluFused(DIM, dropout_rate=0.5, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        out1 = sgf(x, jr.PRNGKey(1), inference=True)
        out2 = sgf(x, jr.PRNGKey(2), inference=True)
        assert jnp.allclose(out1, out2)

    def test_stochastic_in_train_mode(self):
        sgf = SwiGluFused(DIM, dropout_rate=0.9, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        out1 = sgf(x, jr.PRNGKey(1), inference=False)
        out2 = sgf(x, jr.PRNGKey(2), inference=False)
        assert not jnp.allclose(out1, out2)


# get_ffn


class TestGetFfn:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("mlp", Mlp),
            ("swiglu", SwiGlu),
            ("swiglufused", SwiGluFused),
            ("dinohead", DINOHead),
            ("weightnormlinear", WeightNormLinear),
        ],
    )
    def test_string_resolution(self, name, expected):
        assert get_ffn(name) is expected

    def test_class_passthrough(self):
        assert get_ffn(Mlp) is Mlp

    def test_class_passthrough_swiglu(self):
        assert get_ffn(SwiGlu) is SwiGlu

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="unknown module string"):
            get_ffn("nonexistent_ffn")

    def test_returned_class_is_instantiable(self):
        cls = get_ffn("mlp")
        model = cls(DIM, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert model(x, KEY).shape == (SEQLEN, DIM)

    def test_returned_class_instantiable_with_dim_kwargs(self):
        cls = get_ffn("mlp")
        model = cls(DIM, hidden_dim=128, out_dim=32, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert model(x, KEY).shape == (SEQLEN, 32)


# register_ffn


class TestRegisterFfn:
    def test_register_default_name(self):
        import equinox as eqx

        from equimo.layers.ffn import _FFN_REGISTRY, get_ffn, register_ffn

        @register_ffn()
        class CustomFFN(eqx.Module):
            pass

        assert "customffn" in _FFN_REGISTRY
        assert get_ffn("customffn") is CustomFFN

    def test_register_custom_name(self):
        import equinox as eqx

        from equimo.layers.ffn import _FFN_REGISTRY, get_ffn, register_ffn

        @register_ffn(name="MySuperFFN")
        class CustomFFN2(eqx.Module):
            pass

        assert "mysuperffn" in _FFN_REGISTRY
        assert get_ffn("mysuperffn") is CustomFFN2

    def test_register_non_eqx_module(self):
        from equimo.layers.ffn import register_ffn

        with pytest.raises(TypeError, match="must be a subclass of eqx.Module"):

            @register_ffn()
            class NotAModule:
                pass

    def test_register_duplicate_name(self):
        import equinox as eqx

        from equimo.layers.ffn import register_ffn

        @register_ffn()
        class DuplicateFFN(eqx.Module):
            pass

        with pytest.raises(ValueError, match="already registered"):

            @register_ffn(name="DuplicateFFN")
            class AnotherFFN(eqx.Module):
                pass
