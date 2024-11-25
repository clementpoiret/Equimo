import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray


class Downsampler(eqx.Module):
    """A module that performs spatial downsampling using strided convolution.

    This module reduces spatial dimensions (height and width) by a factor of 2
    while optionally increasing the channel dimension. Uses a 3x3 strided
    convolution for downsampling.

    Attributes:
        reduction: Convolutional layer that performs the downsampling
    """

    reduction: eqx.Module

    def __init__(
        self,
        dim: int,
        *,
        key=PRNGKeyArray,
        keep_dim: bool = False,
    ):
        """Initialize the Downsampler.

        Args:
            dim: Number of input channels
            key: PRNG key for initialization
            keep_dim: If True, maintains channel dimension; if False, doubles it
                     (default: False)
        """
        dim_out = dim if keep_dim else 2 * dim
        self.reduction = eqx.nn.Conv(
            num_spatial_dims=2,
            in_channels=dim,
            out_channels=dim_out,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            use_bias=False,
            key=key,
        )

    def __call__(
        self,
        x: Float[Array, "channels height width"],
    ) -> Float[Array, "new_channels new_height new_width"]:
        """Apply downsampling to the input tensor.

        Args:
            x: Input tensor of shape (channels, height, width)

        Returns:
            Downsampled tensor with halved spatial dimensions and potentially
            doubled channels depending on keep_dim parameter
        """
        return self.reduction(x)
