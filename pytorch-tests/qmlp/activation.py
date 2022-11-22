import torch
from torchtyping import TensorType

from .import_library import load_library
load_library()


class FusedActivation(torch.nn.Module):
    """
    Fused activation in a custom CUDA kernel
    """

    def __init__(self, cfg: str):
        """
        Creates the fused activation.

        The activation can either be specified as a name that acts as a key
        in the set of pre-defined activations (see builtin-activations.json).
        Otherwise, it can be specified in-place using the following notation,
        here shown for Sine as an example
        <code>
        {
            "id": "sine",
            "forward": "z = hsin(x)",
            "adjoint": "adjx = __hmul(hcos(x), adjz)"
        }
        </code>
        (This has to be passed as a string to the function for now.
         TODO: accept forward+adjoint as keyword parameters)

        The fused activations only work with half-precision floats!

        :param cfg: the activation name
        """
        super().__init__()
        self._cfg = cfg
        self._activation = torch.classes.qmlp.Activation(cfg)

    def forward(self, input: TensorType["batch", "channels", torch.half]
                ) -> TensorType["batch", "channels", torch.half]:
        """
        Performs a forward pass through the activations with autograd support
        """

        # no grad version:
        # return self._activation.inference(input)

        # with autograd support
        return self._activation.forward(input)

    def __repr__(self):
        return f"FusedActivation(\"{self._cfg}\")"

    def __str__(self):
        return f"FusedActivation(\"{self._cfg}\")"