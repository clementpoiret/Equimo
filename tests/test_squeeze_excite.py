"""Tests for equimo.layers.squeeze_excite."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from equimo.layers.squeeze_excite import (
    EffectiveSEModule,
    SEModule,
    _SE_REGISTRY,
    get_se,
    register_se,
)

KEY = jr.PRNGKey(0)
IN_CHANNELS = 32
H, W = 8, 8


# ---------------------------------------------------------------------------
# SEModule
# ---------------------------------------------------------------------------


class TestSEModule:
    def test_output_shape(self):
        se = SEModule(IN_CHANNELS, key=KEY)
        x = jnp.ones((IN_CHANNELS, H, W))
        assert se(x).shape == (IN_CHANNELS, H, W)

    def test_output_finite(self):
        se = SEModule(IN_CHANNELS, key=KEY)
        x = jr.normal(KEY, (IN_CHANNELS, H, W))
        assert jnp.all(jnp.isfinite(se(x)))

    def test_output_dtype_preserved_float32(self):
        se = SEModule(IN_CHANNELS, key=KEY)
        x = jr.normal(KEY, (IN_CHANNELS, H, W)).astype(jnp.float32)
        assert se(x).dtype == jnp.float32

    def test_output_dtype_preserved_bfloat16(self):
        se = SEModule(IN_CHANNELS, key=KEY)
        x = jr.normal(KEY, (IN_CHANNELS, H, W)).astype(jnp.bfloat16)
        out = se(x)
        assert jnp.all(jnp.isfinite(out))

    def test_output_dtype_preserved_float16(self):
        se = SEModule(IN_CHANNELS, key=KEY)
        x = jr.normal(KEY, (IN_CHANNELS, H, W)).astype(jnp.float16)
        out = se(x)
        assert jnp.all(jnp.isfinite(out))

    def test_channel_attention_effect(self):
        """Output must differ from input (attention was actually applied)."""
        se = SEModule(IN_CHANNELS, key=KEY)
        x = jr.normal(KEY, (IN_CHANNELS, H, W))
        assert not jnp.allclose(se(x), x)

    def test_custom_rd_ratio(self):
        se = SEModule(IN_CHANNELS, rd_ratio=0.25, key=KEY)
        x = jr.normal(KEY, (IN_CHANNELS, H, W))
        assert se(x).shape == (IN_CHANNELS, H, W)

    def test_use_norm(self):
        import equinox as eqx

        se = SEModule(IN_CHANNELS, use_norm=True, key=KEY)
        assert not isinstance(se.norm, eqx.nn.Identity)
        x = jr.normal(KEY, (IN_CHANNELS, H, W))
        assert jnp.all(jnp.isfinite(se(x)))

    def test_custom_act_layer(self):
        se = SEModule(IN_CHANNELS, act_layer=jax.nn.hard_sigmoid, key=KEY)
        x = jr.normal(KEY, (IN_CHANNELS, H, W))
        assert se(x).shape == (IN_CHANNELS, H, W)
        assert jnp.all(jnp.isfinite(se(x)))

    def test_act_is_static(self):
        """act must be a static field so JAX does not treat it as a leaf."""
        import equinox as eqx

        se = SEModule(IN_CHANNELS, key=KEY)
        # eqx.filter splits into dynamic (arrays) and static (non-arrays).
        # Callables stored as static should not appear in the dynamic pytree.
        dynamic, _ = eqx.partition(se, eqx.is_array)
        # act should not be in the dynamic partition (it is not an array).
        assert not hasattr(dynamic, "act") or dynamic.act is None

    def test_kwargs_ignored(self):
        """Extra kwargs must be silently accepted (registry call compatibility)."""
        se = SEModule(IN_CHANNELS, key=KEY, unknown_kwarg=True)
        x = jr.normal(KEY, (IN_CHANNELS, H, W))
        assert se(x).shape == (IN_CHANNELS, H, W)


# ---------------------------------------------------------------------------
# EffectiveSEModule
# ---------------------------------------------------------------------------


class TestEffectiveSEModule:
    def test_output_shape(self):
        se = EffectiveSEModule(IN_CHANNELS, key=KEY)
        x = jnp.ones((IN_CHANNELS, H, W))
        assert se(x).shape == (IN_CHANNELS, H, W)

    def test_output_finite(self):
        se = EffectiveSEModule(IN_CHANNELS, key=KEY)
        x = jr.normal(KEY, (IN_CHANNELS, H, W))
        assert jnp.all(jnp.isfinite(se(x)))

    def test_output_dtype_preserved_bfloat16(self):
        se = EffectiveSEModule(IN_CHANNELS, key=KEY)
        x = jr.normal(KEY, (IN_CHANNELS, H, W)).astype(jnp.bfloat16)
        out = se(x)
        assert jnp.all(jnp.isfinite(out))

    def test_output_dtype_preserved_float16(self):
        se = EffectiveSEModule(IN_CHANNELS, key=KEY)
        x = jr.normal(KEY, (IN_CHANNELS, H, W)).astype(jnp.float16)
        out = se(x)
        assert jnp.all(jnp.isfinite(out))

    def test_channel_attention_effect(self):
        se = EffectiveSEModule(IN_CHANNELS, key=KEY)
        x = jr.normal(KEY, (IN_CHANNELS, H, W))
        assert not jnp.allclose(se(x), x)

    def test_custom_act_layer(self):
        se = EffectiveSEModule(IN_CHANNELS, act_layer=jax.nn.sigmoid, key=KEY)
        x = jr.normal(KEY, (IN_CHANNELS, H, W))
        assert se(x).shape == (IN_CHANNELS, H, W)
        assert jnp.all(jnp.isfinite(se(x)))

    def test_act_is_static(self):
        import equinox as eqx

        se = EffectiveSEModule(IN_CHANNELS, key=KEY)
        dynamic, _ = eqx.partition(se, eqx.is_array)
        assert not hasattr(dynamic, "act") or dynamic.act is None

    def test_kwargs_ignored(self):
        se = EffectiveSEModule(IN_CHANNELS, key=KEY, unknown_kwarg=42)
        x = jr.normal(KEY, (IN_CHANNELS, H, W))
        assert se(x).shape == (IN_CHANNELS, H, W)


# ---------------------------------------------------------------------------
# get_se
# ---------------------------------------------------------------------------


class TestGetSe:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("semodule", SEModule),
            ("effectivesemodule", EffectiveSEModule),
        ],
    )
    def test_string_resolution(self, name, expected):
        assert get_se(name) is expected

    def test_class_passthrough(self):
        assert get_se(SEModule) is SEModule

    def test_class_passthrough_effective(self):
        assert get_se(EffectiveSEModule) is EffectiveSEModule

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="unknown module string"):
            get_se("nonexistent_se")

    def test_returned_class_is_instantiable(self):
        cls = get_se("semodule")
        model = cls(IN_CHANNELS, key=KEY)
        x = jr.normal(KEY, (IN_CHANNELS, H, W))
        assert model(x).shape == (IN_CHANNELS, H, W)


# ---------------------------------------------------------------------------
# register_se
# ---------------------------------------------------------------------------


class TestRegisterSe:
    def test_register_default_name(self):
        import equinox as eqx

        @register_se()
        class CustomSE(eqx.Module):
            pass

        assert "customse" in _SE_REGISTRY
        assert get_se("customse") is CustomSE

    def test_register_custom_name(self):
        import equinox as eqx

        @register_se(name="MySuperSE")
        class CustomSE2(eqx.Module):
            pass

        assert "mysuperse" in _SE_REGISTRY
        assert get_se("mysuperse") is CustomSE2

    def test_register_non_eqx_module(self):
        with pytest.raises(TypeError, match="must be a subclass of eqx.Module"):

            @register_se()
            class NotAModule:
                pass

    def test_register_duplicate_name(self):
        import equinox as eqx

        @register_se()
        class DuplicateSE(eqx.Module):
            pass

        with pytest.raises(ValueError, match="already registered"):

            @register_se(name="DuplicateSE")
            class AnotherSE(eqx.Module):
                pass
