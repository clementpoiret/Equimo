from typing import Optional

import equinox as eqx
import jax
import jax.lax as lax
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray


class DropPath(eqx.Module, strict=True):
    """Applies drop path (stochastic depth).

    Note that this layer behaves differently during training and inference. During
    training then dropout is randomly applied; during inference this layer does nothing.
    """

    p: float
    inference: bool

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

    p: float
    inference: bool

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
            add = jrandom.bernoulli(key, q)
            return jax.lax.cond(add, lambda x, y: x + y, lambda x, y: x, x1, x2)
