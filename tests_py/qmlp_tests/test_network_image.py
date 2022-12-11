import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd
import torch.nn.init
import torch.optim
import imageio
import tqdm

from qmlp.network import FusedNetwork, SimulatedNetwork

def _optimize_image(cfg: str, parent: str, target_file: str, output: str, simulate: bool):
    os.makedirs("../output", exist_ok=True)

    device = torch.device("cuda")

    if simulate:
        network = SimulatedNetwork(cfg, weight_dtype=torch.float32)
    else:
        network = FusedNetwork(cfg, parent, weight_dtype=torch.float32)
    print(network)
    network.to(device=device)

    def eval_with_padding(x):
        B, C = x.shape
        assert C==2
        x = F.pad(x, (0, 14))
        y = network(x)
        return y[:,:3]

    target = imageio.imread(target_file)
    target = (target.astype(np.float32)) / 255.0
    print("Optimize image of shape", target.shape)

    N = target.shape[0] * target.shape[1]
    X, Y = torch.meshgrid(torch.linspace(0, 1, target.shape[0]), torch.linspace(0, 1, target.shape[1]),
                          indexing='xy')
    positions = torch.stack((X, Y), dim=-1).reshape((N, 2)).to(dtype=torch.float32, device=device)
    target_pixels = torch.from_numpy(target).reshape((N, target.shape[2])).to(dtype=torch.float32, device=device)

    # optimize for some epochs
    optim = torch.optim.Adam(network.parameters(), lr=1e-2)
    epochs = 5000
    save_every = 500
    with tqdm.trange(epochs+1) as t:
        for epoch in t:
            optim.zero_grad()
            #prediction = network(positions)
            prediction = eval_with_padding(positions)
            loss = F.mse_loss(prediction, target_pixels)
            loss.backward()
            optim.step()
            t.set_postfix(loss=loss.item())
            if (epoch % save_every) == 0:
                prediction_image = prediction.detach().reshape((target.shape[0], target.shape[1], target.shape[2])).cpu().numpy()
                prediction_image = np.clip(prediction_image*255, 0.0, 255.0).astype(np.uint8)
                imageio.imwrite(os.path.join("../output", output + "-epoch-%d.png" % epoch), prediction_image)


def test_optimize():
    this_folder = os.path.split(__file__)[0]

    configs = [
        "network-2d-relu",
        "network-2d-sine"
        #"network-2d-relu-grid"
    ]

    target_file = os.path.join(this_folder, "../resources/input_art1.jpg")
    for config in configs:
        print("Train with config", config)
        with open(os.path.join(this_folder, f"../resources/{config}.json"), "r") as f:
            cfg = f.read()
        parent = "."
        output = config
        _optimize_image(cfg, parent, target_file, output, simulate=False)

