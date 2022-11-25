import torch
from torchtyping import TensorType

from .import_library import load_library
from .encoding import FusedEncoding
load_library()


class FusedNetwork(torch.nn.Module):
    """
    A fully fused coordinate-MLP combining input encodings,
    linear layers and non-linear activations.
    """

    def __init__(self, cfg: str, parent: str = "."):
        """
        Constructs a new fused network from the given json configuration.
        The second parameter denotes the parent folder of the config, from this folder,
        linked activation specifications are loaded.

        :param cfg: the json configuration string
        :param parent: the parent folder
        """
        super().__init__()
        self._cfg = cfg
        self._parent = parent

        self._network = torch.classes.qmlp.Network(cfg, parent)
        self._num_output_channels: int = self._network.channels_out()

        network_parameter = self._network.create_inference_parameters()
        self._network.initialize_inference_parameters(network_parameter)
        network_parameter.requires_grad_(True)
        self.network_parameter = torch.nn.Parameter(network_parameter)
        print("Number of trainable network parameters:", network_parameter.shape)

        self._num_encodings: int = self._network.num_encodings()
        assert self._num_encodings > 0
        self._max_input_channel = 0
        encoding_parameters = [None] * self._num_encodings
        self._encodings = [None] * self._num_encodings
        for i in range(self._num_encodings):
            e: torch.classes.qmlp.Encoding = self._network.encoding(i)
            print("Encoding at index i:", e.to_json())
            self._encodings[i] = e
            self._max_input_channel = max(self._max_input_channel, e.max_input_channel())
            if e.has_parameters():
                p: torch.Tensor = e.create_inference_parameters()
                p.requires_grad_(True)
                torch.zero_(p)
                encoding_parameters[i] = torch.nn.Parameter(p)
                print("Encoding at index",i,"has parameters of shape", p.shape)
            else:
                dummy = torch.zeros((1,))
                encoding_parameters[i] = torch.nn.Parameter(dummy)
        self.encoding_parameters = torch.nn.ParameterList(encoding_parameters)
        print("Num input channels:", self.num_input_channels())

    @staticmethod
    def MatrixSize() -> int:
        """
        The size of the base matrix multiplication. Layers should be a multiple of this value
        """
        return torch.classes.qmlp.Network.MatrixSize()

    @staticmethod
    def WarpSize() -> int:
        """
        The CUDA warp size. Input batches should be multiple of this value for optimal performance.
        (But this is not required, the inputs are padded if not)
        """
        return torch.classes.qmlp.Network.WarpSize()

    @staticmethod
    def MaxSharedMemoryBytes() -> int:
        """
        The maximal amount of shared memory.
        """
        return torch.classes.qmlp.Network.MaxSharedMemoryBytes()

    def is_parallel_streams(self) -> bool:
        """ True iff parallel streams during backpropagation are enabled """
        return self._network.is_parallel_streams()

    def set_parallel_streams(self, v: bool):
        """
        Enables or disables parallel streams during backpropagation.
        Might speed up the training.
        """
        self._network.set_parallel_streams(v)

    def num_encodings(self) -> int:
        """ Returns the number of encodings prepending the multilayer perceptron """
        return self._network.num_encodings()

    def encoding(self, index: int) -> FusedEncoding:
        """
        Returns the encoding at the given index.
        Use this to access and modify the state of that encoding (if supported)
        """
        return FusedEncoding(self._encodings[index])

    def channels_in(self) -> int:
        """ The expected input channel count """
        return self._network.channels_in()

    def channels_out(self) -> int:
        """ The output channel count produced by the network """
        return self._network.channels_out()

    # TODO: access to weights

    def num_output_channels(self) -> int:
        return self._num_output_channels

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert len(input.shape)==2
        assert input.shape[1] == self.num_input_channels()

        return self._network.forward(input, self.network_parameter, []) #list(self.encoding_parameters))

    def __repr__(self):
        return f"FusedNetwork(\"{self._cfg}\")"

    def __str__(self):
        return f"FusedNetwork(\"{self._cfg}\")"
