import torch
from torchtyping import TensorType

from .import_library import load_library
load_library()


def pullpush(mask: TensorType["batch", "height", "width", float],
             data: TensorType["batch", "channels", "height", "width", float]
             ) -> TensorType["batch", "channels", "height", "width", float]:
    """
    Performs differentiable inpainting via the Pull-Push algorithm.

    All tensors must reside on the GPU and are of type float or double.
    The mask is defined as:
     - 1: non-empty pixel
     - 0: empty pixel
     and any fraction in between.

    :param mask: the mask of shape (Batch, Height, Width)
    :param data: the data of shape (Batch, Channels, Height, Width)
    :output: the inpainted data of shape (Batch, Channels, Height, Width)
    """
    impl = torch.classes.qmlp.utils.pullpush
    return impl(mask, data)