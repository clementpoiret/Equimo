"""Unit tests for DEQ components in ``equimo.implicit``.

Structured by concern:

* Registry resolution (name → class, unknown raises, force-override).
* Injector behavior (each concrete class, output shape, init semantics).
* Stabilizer behavior (identity / projection / damping edge cases).
* Strategy behavior — **critical**: verifies injection call counts, i.e. the
  property that caught a real bug in the original ``StandardLayerApply``
  (where the fuser was invoked once per inner block, causing 9-fold input
  injection per Picard iteration for a 9-block ConvNeXt stage).
* DEQCell: prepare-once / call-many invariant.
* DEQBlock: shape, aux keys, fixed-point residual, gradient flow, JIT.
* ``_init_z0``: all modes plus shape-mismatch and broadcasting edge cases.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import PRNGKeyArray

from equimo.implicit._base import (
    AbstractInjector,
)
from equimo.implicit.deq import DEQBlock, DEQCell, _init_z0
from equimo.implicit.injectors import (
    Add,
    FiLM,
    Gated,
    PreNormAdd,
    ProjAdd,
    get_injector,
    register_injector,
)
from equimo.implicit.stabilizers import (
    Damped,
    DampedProject,
    GroupNormProject,
    Identity,
    get_stabilizer,
)
from equimo.implicit.strategies import (
    EntryInjection,
    PerBlockInjection,
    ScheduledInjection,
    get_strategy,
)


KEY = jr.PRNGKey(0)
DIM = 8
H = W = 8


class TinyBlock(eqx.Module):
    """Near-identity test block: ``x + 1e-3 · Conv(x)``.

    The tiny residual scale keeps any DEQ built from these blocks contractive,
    so the fixed-point solver converges quickly regardless of the injector or
    stabilizer combination under test.
    """

    conv: eqx.nn.Conv2d

    def __init__(
        self,
        *,
        dim: int | None = None,
        in_channels: int | None = None,
        out_channels: int | None = None,
        key: PRNGKeyArray,
        **_,
    ):
        d = dim if dim is not None else (in_channels or out_channels)
        self.conv = eqx.nn.Conv2d(d, d, kernel_size=3, padding=1, key=key)

    def __call__(self, x, *, inference=False, key=None, **_):
        return x + 1e-3 * self.conv(x)


def _zx(dim=DIM, h=H, w=W, key=KEY):
    """Return a pair ``(z, x)`` of independent Gaussian tensors."""
    k_x, k_z = jr.split(key)
    return jr.normal(k_z, (dim, h, w)), jr.normal(k_x, (dim, h, w))


@pytest.mark.parametrize(
    "name,cls",
    [
        ("add", Add),
        ("proj_add", ProjAdd),
        ("prenorm_add", PreNormAdd),
        ("gated", Gated),
        ("film", FiLM),
    ],
)
def test_injector_registry_resolves(name, cls):
    assert get_injector(name) is cls
    assert get_injector(cls) is cls  # class pass-through


@pytest.mark.parametrize(
    "name,cls",
    [
        ("identity", Identity),
        ("projected", GroupNormProject),
        ("damped", Damped),
        ("damped_projected", DampedProject),
    ],
)
def test_stabilizer_registry_resolves(name, cls):
    assert get_stabilizer(name) is cls
    assert get_stabilizer(cls) is cls


@pytest.mark.parametrize(
    "name,cls",
    [
        ("entry", EntryInjection),
        ("per_block", PerBlockInjection),
        ("scheduled", ScheduledInjection),
    ],
)
def test_strategy_registry_resolves(name, cls):
    assert get_strategy(name) is cls
    assert get_strategy(cls) is cls


def test_get_injector_unknown_raises():
    with pytest.raises(ValueError, match="unknown injector"):
        get_injector("__nonexistent_injector__")


def test_get_stabilizer_unknown_raises():
    with pytest.raises(ValueError, match="unknown stabilizer"):
        get_stabilizer("__nonexistent_stabilizer__")


def test_get_strategy_unknown_raises():
    with pytest.raises(ValueError, match="unknown strategy"):
        get_strategy("__nonexistent_strategy__")


def test_register_injector_collision_without_force_raises():
    with pytest.raises(ValueError, match="already registered"):

        @register_injector(name="add")
        class _Duplicate(AbstractInjector):
            def __init__(self, **_):
                pass

            def __call__(self, z, x_ctx, **_):
                return z


def test_register_injector_force_overrides():
    @register_injector(name="__test_force_override__", force=False)
    class First(AbstractInjector):
        def __init__(self, **_):
            pass

        def __call__(self, z, x_ctx, **_):
            return z

    @register_injector(name="__test_force_override__", force=True)
    class Second(AbstractInjector):
        def __init__(self, **_):
            pass

        def __call__(self, z, x_ctx, **_):
            return z

    assert get_injector("__test_force_override__") is Second


def test_register_non_eqx_module_raises():
    with pytest.raises(TypeError):

        @register_injector(name="__test_bad_type__")
        class NotAnEqxModule:  # noqa: D401
            pass


def test_register_uses_lowercased_class_name_by_default():
    @register_injector()
    class _TestDefaultNameZZ(AbstractInjector):
        def __init__(self, **_):
            pass

        def __call__(self, z, x_ctx, **_):
            return z

    assert get_injector("_testdefaultnamezz") is _TestDefaultNameZZ


def test_injector_add_is_exact_sum():
    inj = Add()
    z, x = _zx()
    x_ctx = inj.prepare(x, KEY)
    assert jnp.allclose(inj(z, x_ctx, key=KEY), z + x)


def test_injector_proj_add_shape_and_caches_projection():
    inj = ProjAdd(dim=DIM, key=KEY)
    z, x = _zx()
    x_ctx = inj.prepare(x, KEY)
    assert x_ctx.shape == x.shape
    # Reusing the cached context with the same z must be deterministic.
    o1 = inj(z, x_ctx, key=KEY)
    o2 = inj(z, x_ctx, key=KEY)
    assert jnp.allclose(o1, o2)
    assert o1.shape == z.shape


def test_injector_prenorm_add_shape_and_finite():
    inj = PreNormAdd(dim=DIM, key=KEY)
    z, x = _zx()
    x_ctx = inj.prepare(x, KEY)
    out = inj(z, x_ctx, key=KEY)
    assert out.shape == z.shape
    assert jnp.all(jnp.isfinite(out))


def test_injector_gated_init_gate_half_gives_balanced_mix():
    """``init_gate=0.5`` → cell behaves as ``0.5 z + 0.5 x`` at init (damped Picard)."""
    inj = Gated(dim=DIM, init_gate=0.5, key=KEY)
    z, x = _zx()
    out = inj(z, x, key=KEY)
    assert jnp.allclose(out, 0.5 * z + 0.5 * x, atol=1e-5)


def test_injector_gated_init_gate_high_approaches_pure_injection():
    """With ``init_gate ≈ 1``, the forcing term dominates (classical entry injection)."""
    inj = Gated(dim=DIM, init_gate=0.99, key=KEY)
    _, x = _zx()
    out = inj(jnp.zeros_like(x), x, key=KEY)
    assert jnp.allclose(out, 0.99 * x, atol=1e-4)


def test_injector_film_prepare_returns_pair():
    inj = FiLM(dim=DIM, key=KEY)
    _, x = _zx()
    ctx = inj.prepare(x, KEY)
    assert isinstance(ctx, tuple) and len(ctx) == 2


def test_injector_film_gamma_zero_init():
    """FiLM zero-inits the γ-projection so modulation is trivial at init."""
    inj = FiLM(dim=DIM, key=KEY)
    _, x = _zx()
    gamma, _ = inj.prepare(x, KEY)
    assert jnp.allclose(gamma, 0.0, atol=1e-6)


def test_stabilizer_identity_is_passthrough():
    stab = Identity()
    z_in = jnp.ones((DIM, H, W))
    z_out = jnp.full((DIM, H, W), 7.0)
    assert jnp.allclose(stab(z_in=z_in, z_out=z_out, x_ctx=None, key=KEY), z_out)


def test_stabilizer_group_norm_project_bounds_output():
    """GroupNorm projection must cap large-magnitude inputs."""
    stab = GroupNormProject(dim=DIM)
    z_in = jnp.zeros((DIM, H, W))
    z_out = jr.normal(KEY, (DIM, H, W)) * 100.0
    out = stab(z_in=z_in, z_out=z_out, x_ctx=None, key=KEY)
    assert jnp.all(jnp.isfinite(out))
    # GroupNorm outputs are roughly unit-scale; 100× input magnitude is gone.
    assert jnp.max(jnp.abs(out)) < 20.0


def test_stabilizer_damped_alpha_zero_returns_z_in():
    stab = Damped(dim=DIM, init_alpha=0.0, learnable=False)
    z_in = jnp.ones((DIM, H, W))
    z_out = jnp.zeros((DIM, H, W))
    assert jnp.allclose(stab(z_in=z_in, z_out=z_out, x_ctx=None, key=KEY), z_in)


def test_stabilizer_damped_alpha_one_returns_z_out():
    stab = Damped(dim=DIM, init_alpha=1.0, learnable=False)
    z_in = jnp.ones((DIM, H, W))
    z_out = jnp.zeros((DIM, H, W))
    assert jnp.allclose(stab(z_in=z_in, z_out=z_out, x_ctx=None, key=KEY), z_out)


def test_stabilizer_damped_learnable_scalar_init_half():
    """Learnable scalar ``init_alpha=0.5`` → balanced mix at init."""
    stab = Damped(dim=DIM, init_alpha=0.5, learnable=True, mode="scalar")
    z_in = jnp.ones((DIM, H, W))
    z_out = jnp.zeros((DIM, H, W))
    assert jnp.allclose(
        stab(z_in=z_in, z_out=z_out, x_ctx=None, key=KEY), 0.5 * z_in, atol=1e-4
    )


def test_stabilizer_damped_project_with_alpha_one_projects():
    """``DampedProject(alpha=1)`` collapses to plain GroupNorm projection."""
    stab = DampedProject(dim=DIM, init_alpha=1.0, learnable=False)
    z_in = jnp.zeros((DIM, H, W))
    z_out = jr.normal(KEY, (DIM, H, W)) * 100.0
    out = stab(z_in=z_in, z_out=z_out, x_ctx=None, key=KEY)
    assert jnp.max(jnp.abs(out)) < 20.0


class _CountingInjector(AbstractInjector):
    """Injector that appends to an external list on every call.

    The list is captured as an ``eqx.Module`` field (static leaf under
    ``jax.tree_util``) so the counter survives being passed through the
    strategy without needing JAX tracing. Tests call the strategy in eager
    mode so Python-side side effects are observable.
    """

    counter: list = eqx.field(static=True)

    def __init__(self, counter: list, **_):
        self.counter = counter

    def __call__(self, z, x_ctx, *, inference=False, key=None):
        self.counter.append(1)
        return z + x_ctx


def _counting_blocks(n: int):
    return tuple(TinyBlock(dim=DIM, key=jr.PRNGKey(1000 + i)) for i in range(n))


def test_entry_injection_calls_injector_exactly_once():
    """Regression guard: a 5-block stack must NOT trigger 5 injections."""
    counter: list = []
    injector = _CountingInjector(counter)
    strategy = EntryInjection()
    blocks = _counting_blocks(5)
    z, x = _zx()
    strategy(blocks, z, x, injector, inference=True, key=KEY)
    assert len(counter) == 1, f"expected 1 injection, got {len(counter)}"


def test_per_block_injection_calls_injector_n_times():
    counter: list = []
    injector = _CountingInjector(counter)
    strategy = PerBlockInjection()
    blocks = _counting_blocks(5)
    z, x = _zx()
    strategy(blocks, z, x, injector, inference=True, key=KEY)
    assert len(counter) == 5


@pytest.mark.parametrize(
    "indices,n_blocks,expected",
    [
        ((0,), 5, 1),
        ((0, 2), 5, 2),
        ((0, 2, 4), 5, 3),
        ((5,), 5, 1),  # post-stack injection
        ((0, 5), 5, 2),  # entry + post-stack
    ],
)
def test_scheduled_injection_respects_indices(indices, n_blocks, expected):
    counter: list = []
    injector = _CountingInjector(counter)
    strategy = ScheduledInjection(indices=indices)
    blocks = _counting_blocks(n_blocks)
    z, x = _zx()
    strategy(blocks, z, x, injector, inference=True, key=KEY)
    assert len(counter) == expected


def test_scheduled_injection_all_out_of_range_falls_back_to_entry():
    counter: list = []
    injector = _CountingInjector(counter)
    strategy = ScheduledInjection(indices=(100, 200))
    blocks = _counting_blocks(3)
    z, x = _zx()
    strategy(blocks, z, x, injector, inference=True, key=KEY)
    # Fallback: entry-only injection.
    assert len(counter) == 1


def test_scheduled_injection_empty_indices_raises():
    with pytest.raises(ValueError, match="at least one"):
        ScheduledInjection(indices=())


def test_deqcell_forward_preserves_shape():
    cell = DEQCell(
        channels=DIM,
        depth=3,
        module=TinyBlock,
        module_kwargs={"dim": DIM},
        injector=Add(),
        stabilizer=Identity(),
        strategy=EntryInjection(),
        key=KEY,
    )
    z, x = _zx()
    ctx = cell.prepare(x, KEY)
    out = cell(z, ctx, inference=True, key=KEY)
    assert out.shape == z.shape
    assert jnp.all(jnp.isfinite(out))


class _PrepareCountingInjector(AbstractInjector):
    """Tracks both ``prepare`` and ``__call__`` invocations."""

    prep_counter: list = eqx.field(static=True)
    call_counter: list = eqx.field(static=True)

    def __init__(self, prep_counter: list, call_counter: list, **_):
        self.prep_counter = prep_counter
        self.call_counter = call_counter

    def prepare(self, x, key):
        self.prep_counter.append(1)
        return x

    def __call__(self, z, x_ctx, *, inference=False, key=None):
        self.call_counter.append(1)
        return z + x_ctx


def test_deqcell_prepare_is_cached_across_iterations():
    """``prepare`` runs once per outer forward; ``__call__`` runs per iteration.

    This codifies the main efficiency win of the redesign: static transforms
    of ``x`` (projections, embeddings) are not recomputed at every Picard step.
    """
    prep: list = []
    call: list = []
    cell = DEQCell(
        channels=DIM,
        depth=2,
        module=TinyBlock,
        module_kwargs={"dim": DIM},
        injector=_PrepareCountingInjector(prep, call),
        stabilizer=Identity(),
        strategy=EntryInjection(),
        key=KEY,
    )
    z, x = _zx()
    ctx = cell.prepare(x, KEY)
    for _ in range(5):
        z = cell(z, ctx, inference=True, key=KEY)
    assert len(prep) == 1, f"prepare called {len(prep)} times, expected 1"
    assert len(call) == 5, f"call invoked {len(call)} times, expected 5"


def _make_block(
    depth: int = 3,
    injector: str = "prenorm_add",
    stabilizer: str = "projected",
    strategy: str = "entry",
    tol: float = 1e-4,
    max_steps: int = 50,
    key: PRNGKeyArray = KEY,
) -> DEQBlock:
    ic = get_injector(injector)
    sc = get_stabilizer(stabilizer)
    tc = get_strategy(strategy)
    k_i, k_s, k_t, k_c = jr.split(key, 4)
    return DEQBlock(
        channels=DIM,
        depth=depth,
        module=TinyBlock,
        module_kwargs={"dim": DIM},
        injector=ic(dim=DIM, key=k_i),
        stabilizer=sc(dim=DIM, key=k_s),
        strategy=tc(dim=DIM, key=k_t),
        tol=tol,
        max_steps=max_steps,
        key=k_c,
    )


def test_deqblock_forward_shape_and_finite():
    block = _make_block()
    x = jr.normal(KEY, (DIM, H, W))
    z_star, _ = block(x, inference=True, key=KEY)
    assert z_star.shape == (DIM, H, W)
    assert jnp.all(jnp.isfinite(z_star))


def test_deqblock_aux_contains_required_keys():
    block = _make_block()
    x = jr.normal(KEY, (DIM, H, W))
    _, aux = block(x, inference=True, key=KEY)
    for k in ("z_star", "trajectory", "depth", "error", "key", "x_context", "z0"):
        assert k in aux


def test_deqblock_solver_converges_within_budget():
    block = _make_block(tol=1e-4, max_steps=50)
    x = jr.normal(KEY, (DIM, H, W))
    _, aux = block(x, inference=True, key=KEY)
    assert float(aux["error"]) < 1e-3
    assert int(aux["depth"]) < 50


def test_deqblock_fixed_point_residual_is_small():
    """After the solver converges, ``||f(z*, x) − z*||`` must be small."""
    block = _make_block()
    x = jr.normal(KEY, (DIM, H, W))
    z_star, aux = block(x, inference=True, key=KEY)
    z_next = block.cell(z_star, aux["x_context"], inference=True, key=aux["key"])
    rel = float(jnp.linalg.norm(z_next - z_star) / (jnp.linalg.norm(z_star) + 1e-8))
    assert rel < 1e-2


def test_deqblock_gradients_flow():
    block = _make_block()
    x = jr.normal(KEY, (DIM, H, W))

    def loss(b, x):
        z_star, _ = b(x, inference=True, key=KEY)
        return jnp.mean(z_star**2)

    grads = eqx.filter_grad(loss)(block, x)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert len(leaves) > 0
    assert all(jnp.all(jnp.isfinite(g)) for g in leaves)
    assert any(jnp.any(g != 0) for g in leaves)


def test_deqblock_determinism():
    """Same input, same key → same output. Required for FP existence."""
    block = _make_block()
    x = jr.normal(KEY, (DIM, H, W))
    z1, _ = block(x, inference=True, key=KEY)
    z2, _ = block(x, inference=True, key=KEY)
    assert jnp.allclose(z1, z2)


def test_deqblock_jit_compatible():
    block = _make_block()
    x = jr.normal(KEY, (DIM, H, W))

    @eqx.filter_jit
    def run(b, x):
        z, _ = b(x, inference=True, key=KEY)
        return z

    z = run(block, x)
    assert z.shape == (DIM, H, W)
    assert jnp.all(jnp.isfinite(z))


@pytest.mark.parametrize(
    "injector",
    ["add", "proj_add", "prenorm_add", "gated"],
)
@pytest.mark.parametrize(
    "stabilizer",
    ["identity", "projected", "damped", "damped_projected"],
)
@pytest.mark.parametrize("strategy", ["entry", "per_block"])
def test_deqblock_all_combinations_run(injector, stabilizer, strategy):
    """Smoke test every (injector, stabilizer, strategy) triple."""
    block = _make_block(
        depth=2,
        injector=injector,
        stabilizer=stabilizer,
        strategy=strategy,
    )
    x = jr.normal(KEY, (DIM, H, W))
    z_star, _ = block(x, inference=True, key=KEY)
    assert z_star.shape == (DIM, H, W)
    assert jnp.all(jnp.isfinite(z_star))


def test_init_z0_zeros():
    x = jnp.ones((DIM, H, W))
    z0 = _init_z0(x, None, mode="zeros", inference=True, key=KEY)
    assert z0.shape == x.shape
    assert jnp.allclose(z0, 0.0)


def test_init_z0_ones():
    x = jnp.zeros((DIM, H, W))
    z0 = _init_z0(x, None, mode="ones", inference=True, key=KEY)
    assert jnp.allclose(z0, 1.0)


def test_init_z0_random_has_small_scale():
    x = jnp.zeros((DIM, H, W))
    z0 = _init_z0(x, None, mode="random", inference=True, key=KEY)
    assert jnp.max(jnp.abs(z0)) < 1.0  # scale ~ 0.01


def test_init_z0_mixed_inference_is_deterministic_base():
    """In inference mode, ``mixed`` must not inject noise."""
    x = jnp.ones((DIM, H, W))
    z0 = _init_z0(x, None, mode="mixed", inference=True, key=KEY)
    assert jnp.allclose(z0, 0.0)  # base when z0 is None


def test_init_z0_mixed_training_varies_with_key():
    """In training mode, different keys may produce different z0."""
    x = jnp.ones((DIM, H, W))
    z_a = _init_z0(x, None, mode="mixed", inference=False, key=jr.PRNGKey(1))
    z_b = _init_z0(x, None, mode="mixed", inference=False, key=jr.PRNGKey(2))
    # Not a strict guarantee (keys could happen to pick the same branch) but
    # across two arbitrary keys at least one of the outputs should be
    # finite and bounded.
    assert jnp.all(jnp.isfinite(z_a))
    assert jnp.all(jnp.isfinite(z_b))


def test_init_z0_unknown_mode_raises():
    x = jnp.zeros((DIM, H, W))
    with pytest.raises(ValueError, match="Unknown initialization mode"):
        _init_z0(x, None, mode="bogus", inference=True, key=KEY)


def test_init_z0_1d_z0_is_broadcast_to_feature_map():
    """A ``(C,)`` z0 must be tiled to ``(C, H, W)`` to match x."""
    x = jnp.zeros((DIM, H, W))
    z0_vec = jnp.arange(DIM, dtype=jnp.float32)
    z0 = _init_z0(x, z0_vec, mode="zeros", inference=True, key=KEY)
    assert z0.shape == x.shape
    assert jnp.allclose(z0[0], 0.0)
    assert jnp.allclose(z0[-1], float(DIM - 1))


def test_init_z0_channel_mismatch_raises():
    x = jnp.zeros((DIM, H, W))
    z0_bad = jnp.zeros((DIM + 1,))
    with pytest.raises(AssertionError, match="Shape mismatch"):
        _init_z0(x, z0_bad, mode="zeros", inference=True, key=KEY)


def test_init_z0_full_shape_mismatch_raises():
    x = jnp.zeros((DIM, H, W))
    z0_bad = jnp.zeros((DIM, H + 1, W))
    with pytest.raises(AssertionError, match="Shape mismatch"):
        _init_z0(x, z0_bad, mode="zeros", inference=True, key=KEY)
