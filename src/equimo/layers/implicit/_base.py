from abc import abstractmethod
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

WeightSource = Literal["static", "z", "x", "zx"]
WeightMode = Literal["scalar", "channel"]

# A generic container for input passed through the DEQ loop.
# typically (raw_x, precomputed_features)
InputContext: TypeAlias = tuple[jax.Array, Any]


class _ParameterGenerator(eqx.Module):
    """
    Helper module to generate parameters (logits/weights) either statically or dynamically.
    Used for learned damping, Mann-Halpern coefficients, etc.
    """

    source: WeightSource = eqx.field(static=True)
    mode: WeightMode = eqx.field(static=True)
    out_count: int = eqx.field(static=True)
    dim: int = eqx.field(static=True)

    # If static: array of shape (out_count,) or (out_count, dim, 1, 1)
    # If dynamic: Linear layer mapping pooled_features -> out_count * (1 or dim)
    model: jax.Array | eqx.nn.Linear

    def __init__(
        self,
        dim: int,
        out_count: int,
        source: WeightSource,
        mode: WeightMode,
        init_values: list[float] | jax.Array | float,
        key: PRNGKeyArray | None,
    ):
        self.source = source
        self.mode = mode
        self.out_count = out_count
        self.dim = dim

        # Normalize init_values to array (out_count,)
        if isinstance(init_values, float):
            init_arr = jnp.full((out_count,), init_values)
        else:
            init_arr = jnp.array(init_values)
            if init_arr.ndim == 0:
                init_arr = jnp.repeat(init_arr, out_count)

        assert init_arr.shape[0] == out_count

        if source == "static":
            # Shape logic for broadcasting later
            if mode == "channel":
                # (C, D, 1, 1)
                self.model = jnp.broadcast_to(
                    init_arr[:, None, None, None], (out_count, dim, 1, 1)
                )
            else:
                # (C,)
                self.model = init_arr
        else:
            if key is None:
                raise ValueError(f"Key required for dynamic source '{source}'.")

            in_dim = 0
            if "z" in source:
                in_dim += dim
            if "x" in source:
                in_dim += dim

            linear_out = out_count if mode == "scalar" else out_count * dim
            self.model = eqx.nn.Linear(in_dim, linear_out, key=key)

            # Initialize weights to 0, biases to init_values
            w_zeros = jnp.zeros_like(self.model.weight)
            if mode == "scalar":
                b_init = init_arr
            else:
                # Repeat for channels: [val1, val1, ..., val2, val2...]
                b_init = jnp.repeat(init_arr, dim)

            self.model = eqx.tree_at(
                lambda l: (l.weight, l.bias), self.model, (w_zeros, b_init)
            )

    def __call__(self, z: jax.Array, x: jax.Array) -> jax.Array:
        """Returns tensor of shape (out_count, 1|dim, 1, 1) or (out_count,)"""
        if self.source == "static":
            return self.model

        inputs = []
        if "z" in self.source:
            inputs.append(z)
        if "x" in self.source:
            inputs.append(x)

        concat_input = jnp.concatenate(inputs, axis=0)
        pooled = jnp.mean(concat_input, axis=(-2, -1))

        flat_out = self.model(pooled)

        if self.mode == "scalar":
            # (Out, 1, 1, 1)
            return flat_out[:, None, None, None]
        else:
            # (Out, Dim, 1, 1)
            return flat_out.reshape(self.out_count, self.dim, 1, 1)


class AbstractFuser(eqx.Module):
    """Base class for injecting input `x` into state `z`."""

    def prepare_input(self, x: jax.Array, key: PRNGKeyArray) -> Any:
        """
        Precompute static transformations of x before the loop.
        Returns: A PyTree (e.g. dict or tuple) to be passed as `x` to `__call__`.
        """
        return x

    @abstractmethod
    def __call__(
        self,
        z: jax.Array,
        x: InputContext | jax.Array,
        *,
        inference: bool = False,
        key: PRNGKeyArray,
    ) -> jax.Array:
        pass


class AbstractUpdater(eqx.Module):
    """Base class for combining `z_old` and `z_new` (proposal)."""

    def prepare_input(self, x: jax.Array, key: PRNGKeyArray) -> Any:
        """
        Precompute static transformations of x before the loop.
        Returns: A PyTree to be passed as `x` to `__call__`.
        """
        return x

    @abstractmethod
    def __call__(
        self,
        z_old: jax.Array,
        z_proposed: jax.Array,
        x: InputContext | jax.Array,
        *,
        inference: bool = False,
        key: PRNGKeyArray,
    ) -> jax.Array:
        pass


class AbstractLayerApply(eqx.Module):
    """Decides how a single layer `g`, fuser, and updater interact."""

    @abstractmethod
    def __call__(
        self,
        layer: eqx.Module,
        z: jax.Array,
        x: InputContext,
        fuser: AbstractFuser,
        updater: AbstractUpdater | None,
        *,
        inference: bool,
        key: PRNGKeyArray,
    ) -> jax.Array:
        pass
