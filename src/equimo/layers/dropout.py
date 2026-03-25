from typing import Callable, Optional

import equinox as eqx
import jax
import jax.lax as lax
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

_DROPOUT_REGISTRY: dict[str, type[eqx.Module]] = {}


def register_dropout(
    name: Optional[str] = None,
) -> Callable[[type[eqx.Module]], type[eqx.Module]]:
    """Decorator to dynamically register new dropout modules.

    Why collision checking: Prevents third-party extensions from silently
    overwriting core layers, which can silently corrupt the computational graph.
    """

    def decorator(cls: type[eqx.Module]) -> type[eqx.Module]:
        if not issubclass(cls, eqx.Module):
            raise TypeError(
                f"Registered class must be a subclass of eqx.Module, got {type(cls)}"
            )

        registry_name = name.lower() if name else cls.__name__.lower()

        if registry_name in _DROPOUT_REGISTRY:
            raise ValueError(
                f"Cannot register '{registry_name}'. It is already registered "
                f"to {_DROPOUT_REGISTRY[registry_name]}."
            )

        _DROPOUT_REGISTRY[registry_name] = cls
        return cls

    return decorator


def get_dropout(module: str | type[eqx.Module]) -> type[eqx.Module]:
    """Get a dropout `eqx.Module` class from its registered name.

    This is necessary because configs have to be stringified and stored as
    json files to allow (de)serialization.
    """
    if not isinstance(module, str):
        return module

    module_lower = module.lower()
    if module_lower not in _DROPOUT_REGISTRY:
        raise ValueError(
            f"Got an unknown module string: '{module}'. "
            f"Available modules: {list(_DROPOUT_REGISTRY.keys())}"
        )

    return _DROPOUT_REGISTRY[module_lower]


@register_dropout()
class DropPath(eqx.Module, strict=True):
    """Applies drop path (stochastic depth).

    Note that this layer behaves differently during training and inference. During
    training then dropout is randomly applied; during inference this layer does nothing.
    """

    p: float
    inference: bool

    def __init__(
        self,
        p: float | jax.Array = 0.5,
        inference: bool = False,
    ):
        """**Arguments:**

        - `p`: The fraction of entries to set to zero. (On average.)
        - `inference`: Whether to actually apply dropout at all. If `True` then dropout
            is *not* applied. If `False` then dropout is applied. This may be toggled
            with overridden during [`DropPath.__call__`][].
        """

        if isinstance(p, jax.Array):
            if (_l := len(p)) != 1:
                raise ValueError(f"Got {_l} values for p")
            p = float(p[0])

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
            mask = jr.bernoulli(key, q, shape)
            return x * mask / q


@register_dropout()
class DropPathAdd(eqx.Module, strict=True):
    """Applies drop path (stochastic depth), by adding the second input to the first.

    Note that this layer behaves differently during training and inference. During
    training then dropout is randomly applied; during inference this layer does nothing.
    """

    p: float
    inference: bool

    def __init__(
        self,
        p: float | jax.Array = 0.5,
        inference: bool = False,
    ):
        """**Arguments:**

        - `p`: The fraction of entries to set to zero. (On average.)
        - `inference`: Whether to actually apply dropout at all. If `True` then dropout
            is *not* applied. If `False` then dropout is applied. This may be toggled
            with overridden during [`DropPathAdd.__call__`][].
        """

        if isinstance(p, jax.Array):
            if (_l := len(p)) != 1:
                raise ValueError(f"Got {_l} values for p")
            p = float(p[0])

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
        - `inference`: As per [`DropPathAdd.__init__`][]. If `True` or
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
            add = jr.bernoulli(key, q)
            return lax.cond(add, lambda x, y: x + y, lambda x, y: x, x1, x2)
