import torch
from torchtyping import TensorType
import json
import warnings

from .import_library import load_library
from .encoding import FusedEncoding
from .activation import FusedActivation
load_library()


class FusedNetwork(torch.nn.Module):
    """
    A fully fused coordinate-MLP combining input encodings,
    linear layers and non-linear activations.
    """

    def __init__(self, cfg: str, parent: str = ".", weight_dtype=torch.float16):
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

        self._network = torch.classes.qmlp_cu.Network(cfg, parent)
        self._num_output_channels: int = self._network.channels_out()
        self._num_layers: int = self._network.num_layers()

        weights = self._network.create_inference_parameters()
        self._network.initialize_inference_parameters(weights)
        weights = weights.to(weight_dtype)
        weights.requires_grad_(True)
        self.weights = torch.nn.Parameter(weights)
        print("Number of trainable network parameters:", weights.shape)

        # encodings
        self._num_encodings: int = self._network.num_encodings()
        assert self._num_encodings > 0
        self._max_input_channel = 0
        encoding_parameters = [None] * self._num_encodings
        self._encodings = [None] * self._num_encodings
        for i in range(self._num_encodings):
            e: torch.classes.qmlp_cu.Encoding = self._network.encoding(i)
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
        print("Num input channels:", self.channels_in())

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

    def num_layers(self) -> int:
        return self._num_layers

    def get_weight(self, layer: int, bias: bool, grad: bool) -> torch.Tensor:
        """
        Returns a copy of the weight tensor
        :param layer: the layer index
        :param bias: True -> bias; False -> weight matrix
        :param grad: True -> gradient; False -> inference parameters
        :return: copy of the weight tensor
        """
        p = self.weights.grad if grad else self.weights.data
        t = self._network.parameter_view(layer, bias, p)
        return torch.clone(t.detach())

    def set_weight(self, layer: int, bias: bool, grad: bool, value: torch.Tensor):
        p = self.weights.grad if grad else self.weights.data
        t = self._network.parameter_view(layer, bias, p)
        assert value.shape == t.shape
        assert value.dtype == t.dtype
        t.copy_(value)

    def num_output_channels(self) -> int:
        return self._num_output_channels

    def forward(self, x: TensorType["batch", "in-channels", float]
                ) -> TensorType["batch", "out-channels", float]:
        assert len(x.shape)==2
        assert x.shape[1] == self.channels_in()

        if self.weights.requires_grad and self.weights.dtype == torch.float16:
            warnings.warn("Attempt to train network with 16-bit weights.\n"
                          "While the gradients can be computed, the resulting grad-tensor will also have 16-bit precision and optimizers (e.g. Adam) can't handle those.\n"
                          "Consider enabling 32-bit weights by setting 'weight_dtype=torch.float32' in the constructor.")

        return self._network.forward(x, self.weights, []) #list(self.encoding_parameters))

    def __repr__(self):
        return f"FusedNetwork(\"{self._cfg}\")"

    def __str__(self):
        return f"FusedNetwork(\"{self._cfg}\")"


class SimulatedNetwork(torch.nn.Module):
    """
    Simulates the fused network in regular PyTorch.
    """

    def __init__(self, cfg: str, weight_dtype=torch.float16):
        """
        Constructs a network module that simulates the fused network.
        :param cfg:
        """
        super().__init__()
        cfg = json.loads(cfg)
        self._channels_in = cfg['num_inputs']
        self._channels_out = cfg['num_outputs']
        self._weight_dtype = weight_dtype

        # verify encoding, only 'Identity' allowed in simulation mode
        if len(cfg['encodings']) == 0:
            # implicit identity encoding, all is fine.
            pass
        elif len(cfg['encodings']) == 1:
            e = cfg['encodings'][0]
            if e['id'] != "Identity" or e['start_in'] != 0 or e['n_in'] != self._channels_in:
                raise ValueError("Only identity encoding supported and it must consume all inputs")
        else:
            raise ValueError("Only a single encoding supported")

        # Create layers
        layers = []
        n_in = self._channels_in
        n_out = None
        self._has_bias = []
        for layer in cfg['network']:
            n_out = layer['n_out']
            bias = layer['bias']
            activation = FusedActivation.get_activation(layer['activation'])
            layers.append(torch.nn.Linear(n_in, n_out, bias=bias, dtype=weight_dtype))
            layers.append(activation)
            n_in = n_out
            self._has_bias.append(bias)
        if n_out != self._channels_out:
            raise ValueError("Last layer does not match the number of outputs")
        self.layers = torch.nn.Sequential(*layers)
        self._num_layers = len(cfg['network'])

    def channels_in(self):
        return self._channels_in

    def channels_out(self):
        return self._channels_out

    def num_layers(self) -> int:
        return self._num_layers

    def has_bias(self, layer: int):
        return self._has_bias[layer]

    def get_weight(self, layer: int, bias: bool, grad: bool) -> torch.Tensor:
        """
        Returns a copy of the weight tensor
        :param layer: the layer index
        :param bias: True -> bias; False -> weight matrix
        :param grad: True -> gradient; False -> inference parameters
        :return: copy of the weight tensor
        """
        layer = self.layers[2*layer]
        assert isinstance(layer, torch.nn.Linear)
        t = layer.bias if bias else layer.weight
        return torch.clone(t.grad.detach()) if grad else torch.clone(t.detach())

    def set_weight(self, layer: int, bias: bool, grad: bool, value: torch.Tensor):
        layer = self.layers[2 * layer]
        assert isinstance(layer, torch.nn.Linear)
        t = layer.bias if bias else layer.weight
        if grad:
            t.grad = torch.clone(value.detach())
        else:
            t.copy_(value.detach())

    def forward(self, x: TensorType["batch", "in-channels", float]
                ) -> TensorType["batch", "out-channels", float]:
        return self.layers(x.to(self._weight_dtype)).to(self._weight_dtype)


def copy_torch_to_fused(dst: FusedNetwork, src: SimulatedNetwork, grad: bool = False):
    """
    Copies the weights (or gradients if grad=True) of src to dst
    :param dst: the destination fused network
    :param src: the source simulated network
    :param grad: true if gradients should be copied instead of weights
    """
    assert dst.num_layers() == src.num_layers()
    for layer in range(src.num_layers()):
        w = src.get_weight(layer, False, grad)
        dst.set_weight(layer, False, grad, w)
        if src.has_bias(layer):
            b = src.get_weight(layer, True, grad)
            dst.set_weight(layer, True, grad, b)
