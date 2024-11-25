import warnings
from typing import Optional

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray


class Dropout(eqx.Module, strict=True):
    """Applies dropout.

    Note that this layer behaves differently during training and inference. During
    training then dropout is randomly applied; during inference this layer does nothing.
    """

    # Let's make them static fields, just to avoid possible filtering issues
    p: float = eqx.field(static=True)
    inference: bool = eqx.field(static=True)

    def __init__(
        self,
        p: float = 0.5,
        inference: bool = False,
        *,
        deterministic: Optional[bool] = None,
    ):
        """**Arguments:**

        - `p`: The fraction of entries to set to zero. (On average.)
        - `inference`: Whether to actually apply dropout at all. If `True` then dropout
            is *not* applied. If `False` then dropout is applied. This may be toggled
            with [`equinox.nn.inference_mode`][] or overridden during
            [`equinox.nn.Dropout.__call__`][].
        - `deterministic`: Deprecated alternative to `inference`.
        """

        if deterministic is not None:
            inference = deterministic
            warnings.warn(
                "Dropout(deterministic=...) is deprecated "
                "in favour of Dropout(inference=...)"
            )
        self.p = p
        self.inference = inference

    # Backward compatibility
    @property
    def deterministic(self):
        return self.inference

    @jax.named_scope("eqx.nn.Dropout")
    def __call__(
        self,
        x: Array,
        *,
        key: Optional[PRNGKeyArray] = None,
        inference: Optional[bool] = None,
        deterministic: Optional[bool] = None,
    ) -> Array:
        """**Arguments:**

        - `x`: An any-dimensional JAX array to dropout.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for calculating
            which elements to dropout. (Keyword only argument.)
        - `inference`: As per [`equinox.nn.Dropout.__init__`][]. If `True` or
            `False` then it will take priority over `self.inference`. If `None`
            then the value from `self.inference` will be used.
        - `deterministic`: Deprecated alternative to `inference`.
        """

        if deterministic is not None:
            inference = deterministic
            warnings.warn(
                "Dropout()(deterministic=...) is deprecated "
                "in favour of Dropout()(inference=...)"
            )

        if inference is None:
            inference = self.inference
        if isinstance(self.p, (int, float)) and self.p == 0:
            inference = True
        if inference:
            return x
        elif key is None:
            raise RuntimeError(
                "Dropout requires a key when running in non-deterministic mode."
            )
        else:
            q = 1 - lax.stop_gradient(self.p)
            mask = jrandom.bernoulli(key, q, x.shape)
            return jnp.where(mask, x / q, 0)


class DropPath(eqx.Module, strict=True):
    """Applies drop path (stochastic depth).

    Note that this layer behaves differently during training and inference. During
    training then dropout is randomly applied; during inference this layer does nothing.
    """

    # Let's make them static fields, just to avoid possible filtering issues
    p: float = eqx.field(static=True)
    inference: bool = eqx.field(static=True)

    def __init__(
        self,
        p: float = 0.5,
        inference: bool = False,
    ):
        """**Arguments:**

        - `p`: The fraction of entries to set to zero. (On average.)
        - `inference`: Whether to actually apply dropout at all. If `True` then dropout
            is *not* applied. If `False` then dropout is applied. This may be toggled
            with overridden during [`DropPath.__call__`][].
        """

        self.p = p
        self.inference = inference

    def __call__(
        self,
        x: Array,
        *,
        key: Optional[PRNGKeyArray] = None,
        inference: Optional[bool] = None,
    ) -> Array:
        """**Arguments:**

        - `x`: An any-dimensional JAX array to dropout.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for calculating
            which elements to dropout. (Keyword only argument.)
        - `inference`: As per [`DropPath.__init__`][]. If `True` or
            `False` then it will take priority over `self.inference`. If `None`
            then the value from `self.inference` will be used.
        """

        if inference is None:
            inference = self.inference
        if isinstance(self.p, (int, float)) and self.p == 0:
            inference = True
        if inference:
            return x
        elif key is None:
            raise RuntimeError(
                "DropPath requires a key when running in non-deterministic mode."
            )
        else:
            q = 1 - lax.stop_gradient(self.p)
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            mask = jrandom.bernoulli(key, q, shape)
            return x * mask / q


class DropPathAdd(eqx.Module, strict=True):
    """Applies drop path (stochastic depth), by adding the second input to the first.

    Note that this layer behaves differently during training and inference. During
    training then dropout is randomly applied; during inference this layer does nothing.
    """

    # Let's make them static fields, just to avoid possible filtering issues
    p: float = eqx.field(static=True)
    inference: bool = eqx.field(static=True)

    def __init__(
        self,
        p: float = 0.5,
        inference: bool = False,
    ):
        """**Arguments:**

        - `p`: The fraction of entries to set to zero. (On average.)
        - `inference`: Whether to actually apply dropout at all. If `True` then dropout
            is *not* applied. If `False` then dropout is applied. This may be toggled
            with overridden during [`DropPath.__call__`][].
        """

        self.p = p
        self.inference = inference

    def __call__(
        self,
        x1: Array,
        x2: Array,
        *,
        key: Optional[PRNGKeyArray] = None,
        inference: Optional[bool] = None,
    ) -> Array:
        """**Arguments:**

        - `x1`: An any-dimensional JAX array.
        - `x2`: A x1-dimensional JAX array to stochastically add to x1.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for calculating
            which elements to dropout. (Keyword only argument.)
        - `inference`: As per [`DropPath.__init__`][]. If `True` or
            `False` then it will take priority over `self.inference`. If `None`
            then the value from `self.inference` will be used.
        """

        if inference is None:
            inference = self.inference
        if isinstance(self.p, (int, float)) and self.p == 0:
            inference = True
        if inference:
            return x1 + x2
        elif key is None:
            raise RuntimeError(
                "DropPathAdd requires a key when running in non-deterministic mode."
            )
        else:
            q = 1 - lax.stop_gradient(self.p)
            add = jrandom.bernouilli(key, q)
            return jax.lax.cond(add, lambda x, y: x + y, lambda x, y: x, x1, x2)
