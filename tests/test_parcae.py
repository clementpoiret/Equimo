"""Unit tests for VisionParcae-specific recurrent math.

Structured by concern:

* Scalar helpers and recurrence-depth bounds.
* Injection behavior: exact diagonal discretisation, linear projection, add.
* Step resolution and stochastic sampling contracts.
* Recurrent loop internals: closed-form diagonal dynamics, no-grad suffix split,
  aux diagnostics, JIT compatibility.
* VisionParcae integration points that are specific to Parcae state/readout
  rather than generic image-model smoke coverage.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from equimo.vision.models.parcae import (
    VisionParcae,
    VisionParcaeAdditiveInjection,
    VisionParcaeDiagonalExactZOHInjection,
    VisionParcaeDiagonalInjection,
    VisionParcaeLinearInjection,
    _default_max_recurrence,
    _depth_scaled_std,
    _inverse_softplus,
    _takase_std,
    _to_scalar_i32,
    dynamics_from_alpha,
)


KEY = jr.PRNGKey(0)
DIM = 4
SEQ = 5
NUM_CLASSES = 7
IMG_64 = jr.normal(KEY, (3, 64, 64))


def _tiny_parcae(**kwargs):
    cfg = {
        "img_size": 64,
        "in_channels": 3,
        "dim": DIM,
        "patch_size": 16,
        "num_heads": 1,
        "num_classes": NUM_CLASSES,
        "n_layers_in_prelude": 0,
        "n_layers_in_recurrent_block": 1,
        "n_layers_in_coda": 0,
        "mean_recurrence": 3,
        "mean_backprop_depth": 1,
        "max_recurrence": 4,
        "mlp_ratio": 1.0,
        "prelude_norm": False,
        "state_init": "zero",
        "key": KEY,
    }
    cfg.update(kwargs)
    return VisionParcae(**cfg)


def _coreless(model):
    return eqx.tree_at(lambda m: m.core_block, model, None)


def test_inverse_softplus_round_trips_positive_values():
    x = jnp.asarray([1e-3, 0.1, 1.0, 25.0], dtype=jnp.float32)
    assert jnp.allclose(jax.nn.softplus(_inverse_softplus(x)), x, rtol=1e-5)


def test_std_helpers_match_expected_scaling():
    assert _takase_std(8) == pytest.approx((2.0 / (5.0 * 8.0)) ** 0.5)
    assert _depth_scaled_std(8, depth=4) == pytest.approx(_takase_std(8) / 2.0)
    with pytest.raises(ValueError, match="dim must be positive"):
        _takase_std(0)


def test_default_max_recurrence_reserves_static_scan_budget():
    assert _default_max_recurrence(3, sample_recurrence=False) == 6
    assert _default_max_recurrence(3, sample_recurrence=True) >= 10


def test_dynamics_from_alpha_calibrates_B_init_modes():
    alpha = 0.95
    dt = 1.0
    A_init = -jnp.log(alpha) / dt

    assert dynamics_from_alpha(
        alpha=alpha,
        dt=dt,
        injection_type="diagonal",
        mode="fixed_point",
    ) == pytest.approx((float(A_init), 1.0 - alpha))
    assert dynamics_from_alpha(
        alpha=alpha,
        dt=dt,
        injection_type="diagonal_exact_zoh",
        mode="fixed_point",
    ) == pytest.approx((float(A_init), float(A_init)))
    assert dynamics_from_alpha(
        alpha=alpha,
        dt=dt,
        injection_type="diagonal",
        mode="one_step",
    ) == pytest.approx((float(A_init), 1.0))
    assert dynamics_from_alpha(
        alpha=alpha,
        dt=dt,
        injection_type="diagonal",
        mode="target_depth",
        target_depth=4,
        target_scale=0.5,
    ) == pytest.approx((float(A_init), 0.5 * (1.0 - alpha) / (1.0 - alpha**4)))


def test_dynamics_from_alpha_validates_ambiguous_modes():
    with pytest.raises(ValueError, match="raw mode requires raw_B_init_scale"):
        dynamics_from_alpha(
            alpha=0.95,
            dt=1.0,
            injection_type="diagonal",
            mode="raw",
        )
    with pytest.raises(ValueError, match="target_depth mode requires"):
        dynamics_from_alpha(
            alpha=0.95,
            dt=1.0,
            injection_type="diagonal",
            mode="target_depth",
        )
    with pytest.raises(ValueError, match="only applies to diagonal"):
        dynamics_from_alpha(
            alpha=0.95,
            dt=1.0,
            injection_type="linear",
            mode="fixed_point",
        )


def test_to_scalar_i32_accepts_scalars_and_rejects_bad_inputs():
    assert int(_to_scalar_i32(3, name="steps")) == 3
    assert int(_to_scalar_i32(jnp.asarray(-4), name="steps")) == 0
    with pytest.raises(ValueError, match="steps must be non-negative"):
        _to_scalar_i32(-1, name="steps")
    with pytest.raises(ValueError, match="steps must be a scalar"):
        _to_scalar_i32(jnp.asarray([1, 2]), name="steps")


def test_diagonal_injection_exact_update_and_identity_padded_B():
    inj = VisionParcaeDiagonalInjection(
        input_dim=3,
        state_dim=5,
        dt_init=0.2,
        A_init=2.0,
        B_init_scale=0.5,
    )
    x_t = jnp.ones((2, 5), dtype=jnp.float32)
    e = jnp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float32)

    expected_B = jnp.asarray(
        [
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    injected = e @ expected_B.T
    decay = jnp.exp(-0.2 * 2.0)

    assert jnp.allclose(inj.B, expected_B)
    assert jnp.allclose(inj(x_t, e), x_t * decay + 0.2 * injected)


def test_diagonal_injection_reports_contracting_dynamics():
    inj = VisionParcaeDiagonalInjection(
        input_dim=DIM,
        state_dim=DIM,
        dt_init=0.3,
        A_init=1.7,
    )
    expected = jnp.exp(-0.3 * 1.7)
    assert 0.0 < float(inj.spectral_norm()) < 1.0
    assert jnp.allclose(inj.spectral_norm(), expected)
    assert jnp.allclose(inj.contraction_factor(), expected)


@pytest.mark.parametrize("kwargs", [{"dt_init": 0.0}, {"A_init": 0.0}])
def test_diagonal_injection_rejects_nonpositive_dynamics(kwargs):
    with pytest.raises(ValueError):
        VisionParcaeDiagonalInjection(DIM, DIM, **kwargs)


def test_diagonal_exact_zoh_injection_uses_exact_input_gain():
    inj = VisionParcaeDiagonalExactZOHInjection(
        input_dim=3,
        state_dim=5,
        dt_init=0.2,
        A_init=2.0,
        B_init_scale=0.5,
    )
    x_t = jnp.ones((2, 5), dtype=jnp.float32)
    e = jnp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float32)

    expected_B = jnp.asarray(
        [
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    injected = e @ expected_B.T
    decay = jnp.exp(-0.2 * 2.0)
    input_gain = (1.0 - decay) / 2.0

    assert jnp.allclose(inj.B, expected_B)
    assert jnp.allclose(inj(x_t, e), x_t * decay + input_gain * injected)


def test_diagonal_exact_zoh_matches_diagonal_decay_but_not_euler_input_write():
    exact_zoh = VisionParcaeDiagonalExactZOHInjection(
        input_dim=DIM,
        state_dim=DIM,
        dt_init=0.3,
        A_init=1.7,
    )
    euler = VisionParcaeDiagonalInjection(
        input_dim=DIM,
        state_dim=DIM,
        dt_init=0.3,
        A_init=1.7,
    )
    x_t = jr.normal(KEY, (SEQ, DIM))
    e = jr.normal(jr.fold_in(KEY, 1), (SEQ, DIM))

    expected_decay = jnp.exp(-0.3 * 1.7)
    assert jnp.allclose(exact_zoh.spectral_norm(), expected_decay)
    assert jnp.allclose(exact_zoh.contraction_factor(), expected_decay)
    assert jnp.allclose(
        exact_zoh(x_t, jnp.zeros_like(e)),
        euler(x_t, jnp.zeros_like(e)),
    )
    assert not jnp.allclose(
        exact_zoh(jnp.zeros_like(x_t), e),
        euler(jnp.zeros_like(x_t), e),
    )


@pytest.mark.parametrize("kwargs", [{"dt_init": 0.0}, {"A_init": 0.0}])
def test_diagonal_exact_zoh_injection_rejects_nonpositive_dynamics(kwargs):
    with pytest.raises(ValueError):
        VisionParcaeDiagonalExactZOHInjection(DIM, DIM, **kwargs)


def test_linear_injection_is_exact_concatenated_projection():
    inj = VisionParcaeLinearInjection(2, 3, bias=True, init_std=0.1, key=KEY)
    assert jnp.allclose(inj.adapter.bias, 0.0)

    weight = jnp.arange(15, dtype=jnp.float32).reshape(3, 5) / 10.0
    bias = jnp.asarray([0.25, -0.5, 1.0], dtype=jnp.float32)
    inj = eqx.tree_at(
        lambda m: (m.adapter.weight, m.adapter.bias),
        inj,
        (weight, bias),
    )
    x_t = jnp.asarray([[1.0, 2.0, 3.0], [0.5, 1.5, -1.0]], dtype=jnp.float32)
    e = jnp.asarray([[4.0, 5.0], [2.0, -3.0]], dtype=jnp.float32)
    expected = jnp.concatenate([x_t, e], axis=-1) @ weight.T + bias

    assert jnp.allclose(inj(x_t, e), expected)
    assert jnp.allclose(
        inj.spectral_norm(),
        jnp.linalg.svd(weight[:, :3], compute_uv=False)[0],
    )


def test_additive_injection_is_exact_sum_and_requires_equal_dims():
    inj = VisionParcaeAdditiveInjection(DIM, DIM)
    x_t, e = jr.normal(KEY, (SEQ, DIM)), jr.normal(jr.fold_in(KEY, 1), (SEQ, DIM))
    assert jnp.allclose(inj(x_t, e), x_t + e)
    assert inj.spectral_norm() == pytest.approx(1.0)
    assert inj.contraction_factor() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="recurrent_dim == dim"):
        VisionParcaeAdditiveInjection(input_dim=3, state_dim=4)


def test_step_resolution_fixed_training_and_inference():
    model = _tiny_parcae(mean_recurrence=4, mean_backprop_depth=2, max_recurrence=6)

    no_grad, grad = model._resolve_num_steps(
        num_steps=None,
        num_steps_pair=None,
        inference=False,
        key=KEY,
    )
    assert (int(no_grad), int(grad)) == (2, 2)

    no_grad, grad = model._resolve_num_steps(
        num_steps=None,
        num_steps_pair=None,
        inference=True,
        key=KEY,
    )
    assert (int(no_grad), int(grad)) == (0, 4)


def test_step_resolution_validates_python_ints_and_clips_jax_scalars():
    model = _tiny_parcae(mean_recurrence=2, mean_backprop_depth=1, max_recurrence=3)

    with pytest.raises(ValueError, match="num_steps exceeds max_recurrence"):
        model._resolve_num_steps(
            num_steps=4,
            num_steps_pair=None,
            inference=True,
            key=KEY,
        )
    no_grad, grad = model._resolve_num_steps(
        num_steps=jnp.asarray(9),
        num_steps_pair=None,
        inference=True,
        key=KEY,
    )
    assert (int(no_grad), int(grad)) == (0, 3)

    with pytest.raises(ValueError, match="num_steps_pair exceeds max_recurrence"):
        model._resolve_num_steps(
            num_steps=None,
            num_steps_pair=(2, 2),
            inference=False,
            key=KEY,
        )
    no_grad, grad = model._resolve_num_steps(
        num_steps=None,
        num_steps_pair=jnp.asarray([2, 5]),
        inference=False,
        key=KEY,
    )
    assert (int(no_grad), int(grad)) == (0, 3)


@pytest.mark.parametrize(
    "scheme",
    ["fixed", "poisson-truncated-full", "poisson-full", "poisson-fill"],
)
def test_sampling_schemes_stay_within_static_scan_bound(scheme):
    model = _tiny_parcae(
        mean_recurrence=5,
        mean_backprop_depth=2,
        max_recurrence=8,
        sample_recurrence=True,
        sampling_scheme=scheme,
    )
    no_grad, grad = model.sample_num_steps(KEY, inference=False)
    assert 0 <= int(no_grad) <= 8
    assert 0 <= int(grad) <= 8
    assert int(no_grad + grad) <= 8


def test_sampling_invalid_scheme_raises():
    model = _tiny_parcae(sample_recurrence=True, sampling_scheme="bad-scheme")
    with pytest.raises(ValueError, match="Invalid sampling_scheme"):
        model.sample_num_steps(KEY, inference=False)


@pytest.mark.parametrize("state_init", ["zero", "normal", "embed", "unit", "like-init"])
def test_state_initializers_return_finite_recurrent_states(state_init):
    model = _tiny_parcae(recurrent_dim=6, recurrent_num_heads=1, state_init=state_init)
    e = jr.normal(KEY, (SEQ, DIM))
    h = model._initialize_state(e, key=KEY)
    assert h.shape == (SEQ, 6)
    assert jnp.all(jnp.isfinite(h))
    if state_init == "zero":
        assert jnp.allclose(h, 0.0)
    if state_init == "unit":
        assert jnp.allclose(jnp.mean(h, axis=-1), 0.0, atol=1e-5)
        assert jnp.allclose(jnp.std(h, axis=-1), 1.0, atol=1e-5)


def test_state_initializer_invalid_mode_raises():
    model = _tiny_parcae(state_init="not-a-mode")
    with pytest.raises(ValueError, match="Invalid state_init"):
        model._initialize_state(jnp.ones((SEQ, DIM)), key=KEY)


def test_recurrent_loop_matches_closed_form_diagonal_dynamics_without_core_block():
    model = _coreless(
        _tiny_parcae(
            mean_recurrence=3,
            mean_backprop_depth=3,
            max_recurrence=3,
            dt_init=0.2,
            A_init=1.0,
            B_init_scale=1.0,
        )
    )
    e = jr.normal(KEY, (SEQ, DIM))
    h, aux = model._iterate_recurrent(
        e,
        H=1,
        W=4,
        inference=True,
        key=KEY,
        num_steps=3,
    )

    dt = 0.2
    decay = jnp.exp(-dt)
    expected = dt * (1.0 - decay**3) / (1.0 - decay) * e
    expected_prev = dt * (1.0 - decay**2) / (1.0 - decay) * e

    assert jnp.allclose(h, expected, rtol=1e-5, atol=1e-6)
    assert jnp.allclose(aux["x_recurrent_state_prev"], expected_prev, rtol=1e-5)
    assert jnp.allclose(
        aux["recurrent_residual"],
        jnp.mean(jnp.linalg.norm(expected - expected_prev, axis=-1)),
        rtol=1e-5,
    )


def test_recurrent_loop_matches_closed_form_exact_zoh_without_core_block():
    model = _coreless(
        _tiny_parcae(
            injection_type="diagonal_exact_zoh",
            mean_recurrence=3,
            mean_backprop_depth=3,
            max_recurrence=3,
            dt_init=0.2,
            A_init=2.0,
            B_init_scale=1.0,
        )
    )
    e = jr.normal(KEY, (SEQ, DIM))
    h, aux = model._iterate_recurrent(
        e,
        H=1,
        W=4,
        inference=True,
        key=KEY,
        num_steps=3,
    )

    decay = jnp.exp(-0.2 * 2.0)
    input_gain = (1.0 - decay) / 2.0
    expected = input_gain * (1.0 - decay**3) / (1.0 - decay) * e
    expected_prev = input_gain * (1.0 - decay**2) / (1.0 - decay) * e

    assert isinstance(model.adapter, VisionParcaeDiagonalExactZOHInjection)
    assert jnp.allclose(h, expected, rtol=1e-5, atol=1e-6)
    assert jnp.allclose(aux["x_recurrent_state_prev"], expected_prev, rtol=1e-5)
    assert jnp.allclose(
        aux["recurrent_residual"],
        jnp.mean(jnp.linalg.norm(expected - expected_prev, axis=-1)),
        rtol=1e-5,
    )


def test_alpha_init_one_step_mode_initializes_euler_pass_through_write():
    model = _coreless(
        _tiny_parcae(
            mean_recurrence=1,
            mean_backprop_depth=1,
            max_recurrence=1,
            dt_init=1.0,
            alpha_init=0.95,
            B_init_mode="one_step",
        )
    )
    e = jr.normal(KEY, (SEQ, DIM))
    h, _ = model._iterate_recurrent(
        e,
        H=1,
        W=4,
        inference=True,
        key=KEY,
        num_steps=1,
    )

    assert model.B_init_mode == "one_step"
    assert jnp.allclose(jnp.diag(model.adapter.B), 1.0)
    assert jnp.allclose(h, e, rtol=1e-5, atol=1e-6)


def test_alpha_init_target_depth_mode_initializes_exact_zoh_target_write():
    target_depth = 4
    model = _coreless(
        _tiny_parcae(
            injection_type="diagonal_exact_zoh",
            mean_recurrence=target_depth,
            mean_backprop_depth=target_depth,
            max_recurrence=target_depth,
            dt_init=1.0,
            alpha_init=0.95,
            B_init_mode="target_depth",
            B_init_target_depth=target_depth,
        )
    )
    e = jr.normal(KEY, (SEQ, DIM))
    h, _ = model._iterate_recurrent(
        e,
        H=1,
        W=4,
        inference=True,
        key=KEY,
        num_steps=target_depth,
    )

    assert model.B_init_mode == "target_depth"
    assert model.B_init_target_depth == target_depth
    assert isinstance(model.adapter, VisionParcaeDiagonalExactZOHInjection)
    assert jnp.allclose(h, e, rtol=1e-5, atol=1e-6)


def test_alpha_init_rejects_ambiguous_B_init_configuration():
    with pytest.raises(ValueError, match="requires B_init_mode='raw'"):
        _tiny_parcae(
            alpha_init=0.95,
            B_init_mode="one_step",
            B_init_scale=1.0,
        )
    with pytest.raises(ValueError, match="requires alpha_init"):
        _tiny_parcae(B_init_mode="target_depth", B_init_target_depth=2)


def test_recurrent_loop_zero_steps_returns_initial_state():
    model = _coreless(_tiny_parcae(max_recurrence=3))
    e = jr.normal(KEY, (SEQ, DIM))
    h, aux = model._iterate_recurrent(
        e,
        H=1,
        W=4,
        inference=True,
        key=KEY,
        num_steps=0,
    )
    assert jnp.allclose(h, 0.0)
    assert int(aux["num_steps"]) == 0
    assert jnp.allclose(aux["recurrent_residual"], 0.0)


def test_recurrent_loop_no_grad_prefix_blocks_prefix_gradient():
    model = _coreless(
        _tiny_parcae(
            mean_recurrence=3,
            mean_backprop_depth=1,
            max_recurrence=3,
            dt_init=0.2,
            A_init=1.0,
        )
    )
    e = jr.normal(KEY, (SEQ, DIM))

    def loss_with_pair(e_, pair):
        h, _ = model._iterate_recurrent(
            e_,
            H=1,
            W=4,
            inference=False,
            key=KEY,
            num_steps_pair=pair,
        )
        return jnp.sum(h)

    grad_suffix_only = jax.grad(lambda e_: loss_with_pair(e_, (2, 1)))(e)
    grad_all_steps = jax.grad(lambda e_: loss_with_pair(e_, (0, 3)))(e)

    decay = jnp.exp(-0.2)
    assert jnp.allclose(grad_suffix_only, jnp.full_like(e, 0.2), rtol=1e-5)
    assert jnp.allclose(
        grad_all_steps,
        jnp.full_like(e, 0.2 * (1.0 + decay + decay**2)),
        rtol=1e-5,
    )


def test_forward_features_aux_and_readout_shapes():
    model = _tiny_parcae(dim=8, recurrent_dim=8, num_heads=2, recurrent_num_heads=2)
    fwd = model.forward_features(IMG_64, key=KEY, inference=True)
    y = model.readout(fwd["x_recurrent_state"], key=KEY, inference=True)

    assert fwd["x_norm_cls_token"].shape == (8,)
    assert fwd["x_norm_patchtokens"].shape == (16, 8)
    assert fwd["x_recurrent_state"].shape == (17, 8)
    assert fwd["x_projected_recurrent"].shape == (17, 8)
    assert y.shape == (NUM_CLASSES,)
    assert jnp.all(jnp.isfinite(y))


def test_cls_patch_mean_global_pool_forward_shape():
    model = _tiny_parcae(global_pool="cls_patch_mean")
    y = model(IMG_64, key=KEY, inference=True)

    assert model.head.in_features == 2 * DIM
    assert y.shape == (NUM_CLASSES,)
    assert jnp.all(jnp.isfinite(y))


def test_C_identity_init_sets_identity_weight_and_zero_bias():
    model = _tiny_parcae(C_init="identity", C_bias=True)

    assert model.C_init == "identity"
    assert jnp.allclose(model.C.weight, jnp.eye(DIM, dtype=model.C.weight.dtype))
    assert model.C.bias is not None
    assert jnp.allclose(model.C.bias, 0.0)


def test_C_identity_init_respects_no_bias_and_pads_mismatched_dims():
    model = _tiny_parcae(C_init="identity", C_bias=False)
    assert model.C.bias is None

    wider_state = _tiny_parcae(C_init="identity", recurrent_dim=DIM + 2)
    expected_wider = jnp.zeros_like(wider_state.C.weight)
    expected_wider = expected_wider.at[jnp.arange(DIM), jnp.arange(DIM)].set(1.0)
    assert wider_state.C.weight.shape == (DIM, DIM + 2)
    assert jnp.allclose(wider_state.C.weight, expected_wider)

    narrower_state = _tiny_parcae(C_init="identity", recurrent_dim=DIM - 1)
    expected_narrower = jnp.zeros_like(narrower_state.C.weight)
    expected_narrower = expected_narrower.at[
        jnp.arange(DIM - 1), jnp.arange(DIM - 1)
    ].set(1.0)
    assert narrower_state.C.weight.shape == (DIM, DIM - 1)
    assert jnp.allclose(narrower_state.C.weight, expected_narrower)


def test_weight_decay_mask_excludes_parcae_dynamics_and_readout_projection():
    model = _tiny_parcae()
    mask = model.weight_decay_mask()
    assert mask.C.weight is False
    assert mask.adapter.A_log is False
    assert mask.adapter.dt_bias is False
    assert mask.adapter.B is False


def test_recurrent_loop_filter_jit_accepts_dynamic_step_scalar():
    model = _coreless(_tiny_parcae(max_recurrence=3))
    e = jr.normal(KEY, (SEQ, DIM))

    @eqx.filter_jit
    def run(m, e_, steps):
        h, aux = m._iterate_recurrent(
            e_,
            H=1,
            W=4,
            inference=True,
            key=KEY,
            num_steps=steps,
        )
        return h, aux["num_steps"]

    h, steps = run(model, e, jnp.asarray(9))
    assert h.shape == e.shape
    assert int(steps) == 3
    assert jnp.all(jnp.isfinite(h))
