# Draws inspiration from Timm's ModelEmaV2
# <https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/model_ema.py#L83>

from copy import deepcopy

import equinox as eqx
import jax


class EmaModel:
    """
    Exponential Moving Average (EMA) model wrapper.

    This class implements an EMA model, which maintains a moving average of the weights
    of another model. This can be useful for model smoothing and improving generalization.

    Example usage can be Sharpness-Aware Training for Free[1].

    Attributes:
        model (nnx.Module): The underlying model to apply EMA to.
        decay (float): The decay rate for the moving average. Default is 0.9999.

    References:
        [1] Du, et al. Sharpness-Aware Training for Free (2022). https://arxiv.org/abs/2205.14083
    """

    model: eqx.Module
    decay: float

    def __init__(self, model: eqx.Module, decay: float = 0.9999):
        self.model = deepcopy(model)
        self.decay = decay
        self.ema_params = None

    def features(self, *args, **kwargs):
        """
        Get the underlying model's features.

        This method relies on the implementation of the `features()` function
        of the underlying model.

        Args:
            *args: Positional arguments to pass to the underlying model.
            **kwargs: Keyword arguments to pass to the underlying model.

        Returns:
            The output of the underlying model.
        """
        return self.model.features(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """
        Call the underlying model.

        This method allows the EMA model to be used as a drop-in replacement for the original model.

        Args:
            *args: Positional arguments to pass to the underlying model.
            **kwargs: Keyword arguments to pass to the underlying model.

        Returns:
            The output of the underlying model.
        """
        return self.model(*args, **kwargs)

    def initialize_ema(self, params):
        """
        Initialize the EMA parameters.

        This method creates a copy of the initial model parameters to start the EMA process.

        Args:
            params: The initial parameters of the model.
        """
        self.ema_params = jax.tree.map(lambda x: x.copy(), params)

    def update(self, params):
        """
        Update the EMA parameters.

        This method updates the EMA parameters based on the current model parameters.

        Args:
            params: The current parameters of the model.
        """
        if self.ema_params is None:
            self.initialize_ema(params)
        else:
            self.ema_params = jax.tree.map(
                lambda ema, new: self.decay * ema + (1.0 - self.decay) * new,
                self.ema_params,
                params,
            )

        self.model = eqx.apply_updates(self.model, self.ema_params)

    def set(self, params):
        """
        Set the EMA parameters directly.

        This method allows for manual setting of the EMA parameters.

        Args:
            params: The parameters to set as the new EMA parameters.
        """
        self.ema_params = jax.tree.map(lambda x: x.copy(), params)

    def get_ema_params(self):
        """
        Get the current EMA parameters.

        Returns:
            The current EMA parameters.
        """
        return self.ema_params
