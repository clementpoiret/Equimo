from typing import Any

import equinox as eqx
import jax


@jax.jit
def update_ema_model(
    ema_model: eqx.Module, current_params: Any, decay: float = 0.9999
) -> eqx.Module:
    """Updates EMA model with new parameters using exponential moving average.

    Args:
        ema_model: The EMA model to update
        current_params: Current parameters from the training model
        decay: EMA decay rate

    Returns:
        Updated EMA model
    """
    # Get current EMA parameters
    ema_params = eqx.filter(ema_model, eqx.is_array)

    # Update EMA parameters
    new_ema_params = jax.tree_map(
        lambda ema, new: decay * ema + (1.0 - decay) * new, ema_params, current_params
    )

    return eqx.apply_updates(ema_model, new_ema_params)
