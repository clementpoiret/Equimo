from typing import Optional

import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray

from equimo.layers.dropout import DropPathAdd


class Residual(eqx.Module):
    """A wrapper module that adds a residual connection with optional drop path.

    This module wraps any other module and adds a residual (skip) connection around it.
    It also includes drop path regularization which stochastically drops the residual
    path during training. The computation flow is:
    input -> [main branch: module] + [residual branch: identity with drop path] -> output

    Attributes:
        module: The module to wrap with a residual connection
        drop_path: DropPath module for residual connection regularization
    """

    module: eqx.Module
    drop_path: DropPathAdd

    def __init__(
        self,
        module: eqx.Module,
        drop_path: float = 0,
    ):
        """Initialize the Residual wrapper.

        Args:
            module: The module to wrap with a residual connection
            drop_path: Drop path rate (probability of dropping the residual connection)
                      (default: 0)
        """
        self.module = module
        self.drop_path = DropPathAdd(drop_path)

    def __call__(
        self,
        x: Float[Array, "..."],
        key: PRNGKeyArray,
        pass_args: bool = False,
        inference: Optional[bool] = None,
    ) -> Float[Array, "..."]:
        """Forward pass of the residual block.

        Args:
            x: Input tensor of any shape
            enable_dropout: Whether to enable dropout during training
            key: PRNG key for randomness
            pass_args: Whether to pass enable_dropout and key to the wrapped module
                      (default: False)

        Returns:
            Output tensor with same shape as input, combining the module output
            with the residual connection through drop path
        """
        if pass_args:
            x2 = self.module(x, inference=inference, key=key)
        else:
            x2 = self.module(x)

        return self.drop_path(
            x,
            x2,
            inference=inference,
            key=key,
        )
