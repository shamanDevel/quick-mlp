import os
import numpy as np
import torch
import torch.nn.functional
import torch.autograd
import torch.nn.init
import torch.optim
import imageio

from import_library import load_library
load_library()

class FusedNetwork(torch.nn.Module):

    def __init__(self, cfg: str, parent: str):
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

    def num_input_channels(self) -> int:
        return self._max_input_channel+1

    def num_output_channels(self) -> int:
        return self._num_output_channels

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert len(input.shape)==2
        assert input.shape[1] == self.num_input_channels()

        return self._network.forward(input, self.network_parameter, []) #list(self.encoding_parameters))


def _optimize_image(cfg: str, parent: str, target_file: str, output: str):
    os.makedirs("output", exist_ok=True)

    device = torch.device("cuda")

    network = FusedNetwork(cfg, parent)
    print(network)
    network.to(device=device)

    target = imageio.imread(target_file)
    print("Optimize image of shape", target.shape)

    N = target.shape[0] * target.shape[1]
    X, Y = torch.meshgrid(torch.arange(target.shape[0]), torch.arange(target.shape[1]),
                          indexing='xy')
    positions = torch.stack((X, Y), dim=-1).reshape((N, 2)).to(dtype=torch.float32, device=device)
    target_pixels = torch.from_numpy(target).reshape((N, target.shape[2])).to(dtype=torch.float32, device=device)

    # optimize for some epochs
    optim = torch.optim.Adam(network.parameters(), lr=1e-3)
    epochs = 100
    save_every = 10
    for epoch in range(epochs+1):
        optim.zero_grad()
        prediction = network(positions)
        loss = torch.nn.functional.mse_loss(prediction, target_pixels)
        loss.backward()
        optim.step()
        print(f"Epoch {epoch} -> loss={loss.item()}")
        if (epoch % save_every) == 0:
            prediction_image = prediction.detach().reshape((target.shape[0], target.shape[1], target.shape[2])).cpu().numpy()
            prediction_image = np.clip(prediction_image*255, 0.0, 255.0).astype(np.uint8)
            imageio.imwrite(os.path.join("output", output+"-epoch-%d.png"%epoch), prediction_image)


def test_optimize_relu():
    this_folder = os.path.split(__file__)[0]
    with open(os.path.join(this_folder, "network-2d-relu.json"), "r") as f:
        cfg = f.read()
    parent = "."
    target_file = os.path.join(this_folder, "input_art1.jpg")
    output = "network-relu"
    _optimize_image(cfg, parent, target_file, output)


if __name__ == '__main__':
    test_optimize_relu()
