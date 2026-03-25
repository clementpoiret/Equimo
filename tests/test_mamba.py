"""Tests for equimo.layers.mamba."""

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

from equimo.layers.mamba import Mamba2Mixer, get_mixer, register_mixer
from equimo.layers.norm import RMSNormGated

# Shared fixtures

KEY = jr.PRNGKey(0)
SEQLEN = 16
DIM = 64  # dim=64, expand=2 → d_inner=128, head_dim=64 → n_heads=2


# Mamba2Mixer


class TestMamba2Mixer:
    def test_output_shape(self):
        mixer = Mamba2Mixer(DIM, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert mixer(x, key=KEY).shape == (SEQLEN, DIM)

    def test_output_finite(self):
        mixer = Mamba2Mixer(DIM, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert jnp.all(jnp.isfinite(mixer(x, key=KEY)))

    def test_bfloat16_input_finite(self):
        """bfloat16 inputs must not produce NaN/Inf despite exp/softplus ops."""
        mixer = Mamba2Mixer(DIM, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM)).astype(jnp.bfloat16)
        assert jnp.all(jnp.isfinite(mixer(x, key=KEY)))

    def test_float16_input_finite(self):
        """float16 inputs must not produce NaN/Inf despite exp/softplus ops."""
        mixer = Mamba2Mixer(DIM, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM)).astype(jnp.float16)
        assert jnp.all(jnp.isfinite(mixer(x, key=KEY)))

    def test_bfloat16_params_finite(self):
        """Low-precision params (A_log in bfloat16) must not cause NaN/Inf via exp."""
        mixer = Mamba2Mixer(DIM, key=KEY)
        mixer = eqx.tree_at(lambda m: m.A_log, mixer, mixer.A_log.astype(jnp.bfloat16))
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert jnp.all(jnp.isfinite(mixer(x, key=KEY)))

    def test_deterministic(self):
        """Mamba2Mixer has no stochastic ops; output must not vary across keys."""
        mixer = Mamba2Mixer(DIM, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        out1 = mixer(x, key=jr.PRNGKey(1))
        out2 = mixer(x, key=jr.PRNGKey(2))
        assert jnp.allclose(out1, out2)

    def test_inference_flag_no_effect(self):
        """inference flag has no effect (no dropout); output must be identical."""
        mixer = Mamba2Mixer(DIM, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        out_train = mixer(x, key=KEY, inference=False)
        out_infer = mixer(x, key=KEY, inference=True)
        assert jnp.allclose(out_train, out_infer)

    def test_d_inner_not_multiple_of_head_dim_raises(self):
        """d_inner = dim * expand must be divisible by head_dim."""
        with pytest.raises(ValueError, match="head_dim"):
            # dim=64, expand=2 → d_inner=128; 128 % 48 != 0
            Mamba2Mixer(DIM, expand=2, head_dim=48, key=KEY)

    def test_param_shapes(self):
        expand = 2
        head_dim = 64
        mixer = Mamba2Mixer(DIM, expand=expand, head_dim=head_dim, key=KEY)
        expected_n_heads = (DIM * expand) // head_dim
        assert mixer.A_log.shape == (expected_n_heads,)
        assert mixer.D.shape == (expected_n_heads,)
        assert mixer.dt_bias.shape == (expected_n_heads,)

    def test_D_initial_ones(self):
        mixer = Mamba2Mixer(DIM, key=KEY)
        assert jnp.all(mixer.D == 1.0)

    def test_d_inner_stored(self):
        mixer = Mamba2Mixer(DIM, expand=2, key=KEY)
        assert mixer.d_inner == DIM * 2

    def test_n_heads_stored(self):
        mixer = Mamba2Mixer(DIM, expand=2, head_dim=64, key=KEY)
        assert mixer.n_heads == (DIM * 2) // 64

    def test_custom_expand(self):
        mixer = Mamba2Mixer(DIM, expand=4, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert mixer(x, key=KEY).shape == (SEQLEN, DIM)

    def test_custom_head_dim(self):
        # dim=64, expand=2 → d_inner=128; head_dim=32 → n_heads=4
        mixer = Mamba2Mixer(DIM, head_dim=32, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert mixer(x, key=KEY).shape == (SEQLEN, DIM)

    def test_custom_n_groups(self):
        # n_heads=2, n_groups=2 — valid since h % g == 0
        mixer = Mamba2Mixer(DIM, n_groups=2, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert mixer(x, key=KEY).shape == (SEQLEN, DIM)

    def test_no_bias(self):
        mixer = Mamba2Mixer(DIM, use_bias=False, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert jnp.all(jnp.isfinite(mixer(x, key=KEY)))

    def test_with_bias(self):
        mixer = Mamba2Mixer(DIM, use_bias=True, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert jnp.all(jnp.isfinite(mixer(x, key=KEY)))

    def test_norm_layer_default_is_layernorm(self):
        mixer = Mamba2Mixer(DIM, key=KEY)
        assert isinstance(mixer.norm, eqx.nn.LayerNorm)

    def test_norm_layer_rmsnormgated(self):
        """RMSNormGated path must produce finite output."""
        mixer = Mamba2Mixer(DIM, norm_layer=RMSNormGated, key=KEY)
        assert isinstance(mixer.norm, RMSNormGated)
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert jnp.all(jnp.isfinite(mixer(x, key=KEY)))

    def test_drop_path_kwarg_absorbed(self):
        """BlockChunk passes drop_path; **kwargs must absorb it silently."""
        mixer = Mamba2Mixer(DIM, drop_path=0.1, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert mixer(x, key=KEY).shape == (SEQLEN, DIM)

    def test_variable_seqlens(self):
        """Mixer must handle different sequence lengths without recompilation."""
        mixer = Mamba2Mixer(DIM, key=KEY)
        for seqlen in [8, 16, 32]:
            x = jr.normal(KEY, (seqlen, DIM))
            assert mixer(x, key=KEY).shape == (seqlen, DIM)


# get_mixer


class TestGetMixer:
    def test_string_resolution(self):
        assert get_mixer("mamba2mixer") is Mamba2Mixer

    def test_class_passthrough(self):
        assert get_mixer(Mamba2Mixer) is Mamba2Mixer

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="unknown module string"):
            get_mixer("nonexistent_mixer")

    def test_returned_class_is_instantiable(self):
        cls = get_mixer("mamba2mixer")
        mixer = cls(DIM, key=KEY)
        x = jr.normal(KEY, (SEQLEN, DIM))
        assert mixer(x, key=KEY).shape == (SEQLEN, DIM)


# register_mixer


class TestRegisterMixer:
    def test_register_default_name(self):
        from equimo.layers.mamba import _MIXER_REGISTRY

        @register_mixer()
        class CustomMixer(eqx.Module):
            pass

        assert "custommixer" in _MIXER_REGISTRY
        assert get_mixer("custommixer") is CustomMixer

    def test_register_custom_name(self):
        from equimo.layers.mamba import _MIXER_REGISTRY

        @register_mixer(name="MySpecialMixer")
        class CustomMixer2(eqx.Module):
            pass

        assert "myspecialmixer" in _MIXER_REGISTRY
        assert get_mixer("myspecialmixer") is CustomMixer2

    def test_register_non_eqx_module_raises(self):
        with pytest.raises(TypeError, match="must be a subclass of eqx.Module"):

            @register_mixer()
            class NotAModule:
                pass

    def test_register_duplicate_name_raises(self):
        @register_mixer()
        class DuplicateMixer(eqx.Module):
            pass

        with pytest.raises(ValueError, match="already registered"):

            @register_mixer(name="DuplicateMixer")
            class AnotherMixer(eqx.Module):
                pass
