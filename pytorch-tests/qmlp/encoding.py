import torch
from torchtyping import TensorType
from typing import overload

from .import_library import load_library
load_library()


class FusedEncoding(torch.nn.Module):
    """
    Fused encoding in a custom CUDA kernel
    """

    @overload
    def __init__(self, cfg: str):
        """
        Creates the encoding from the json-string given in 'cfg'.
        The config always contains the keys
         - id: identifier of the encoding class
         - start_in: start channel
         - n_in: number of input channels.
        The encoding will then ingest the input channels [start_in: start_in+n_in].
        The number of outputs depends on the encoding.

        Examples:
        <code>
        {
          "id": "Identity",
          "start_in": 0,
          "n_in": 2
        }
        </code>

        <code>
        {
            "id": "HashGrid",
            "start_in": 0,
            "n_in": 1,
            "n_levels": 1,
            "n_features_per_level": 16,
            "log2_hashmap_size": -1,
            "min_resolution": 32,
            "max_resolution": 16,
            "combination_mode": "add",
            "bounding_box_min": [-1],
            "bounding_box_size": [2]
        }
        </code>

        """
        ...

    @overload
    def __init__(self, impl: torch.classes.qmlp.Encoding):
        """
        Wraps the encoding directly.
        This is used by the fused network
        """
        ...

    def __init__(self, cfg_or_impl):
        super().__init__()
        if isinstance(cfg_or_impl, str):
            self._cfg = cfg_or_impl
            self._encoding = torch.classes.qmlp.Encoding(cfg_or_impl)
        elif isinstance(cfg_or_impl, torch.classes.qmlp.Encoding):
            self._cfg = cfg_or_impl.to_json()
            self._encoding = cfg_or_impl
        self._max_input_channel: int = self._encoding.max_input_channel()
        self._num_output_channels : int= self._encoding.num_output_channels()
        self._has_parameters: bool = self._encoding.has_parameters()

    def max_input_channel(self) -> int:
        """The maximal input channel consumed"""
        return self._max_input_channel

    def num_output_channels(self) -> int:
        """The number of output channels"""
        return self._num_output_channels

    def has_parameters(self) -> bool:
        """
        Returns true iff this encoding contains trainable parameters.
        If yes, create_parameter_tensor() can be used to create a tensor
        of suitable size and dtype
        """
        return self._has_parameters

    def create_parameter_tensor(self) -> torch.Tensor:
        """
        Creates the tensor for storing the encoding parametesr
        """
        assert self.has_parameters()
        t: torch.Tensor = self._encoding.create_inference_parameters()
        torch.nn.init.normal_(t)
        return t

    def forward(self, input: TensorType["batch", "channels", float],
                parameters: torch.Tensor=None
                ) -> TensorType["batch", "out-channels", torch.half]:
        """
        Performs a forward pass through the encoding with autograd support.
        Note that the inputs are assumed to be of type 'float',
        the outputs, however, will be of type 'half'.

        If the encoding has parameters (see  has_parameters() ),
        the parameter tensor has to be passed as well.
        """

        # no grad version:
        # return self._encoding.inference(input)

        # with autograd support
        if parameters is None:
            return self._encoding.forward(input)
        else:
            return self._encoding.forward_with_parameter(input, parameters)

    def __repr__(self):
        return f'FusedEncoding("""{self._cfg}""")'

    def __str__(self):
        return f'FusedEncoding("""{self._cfg}""")'

