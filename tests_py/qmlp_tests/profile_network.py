"""
Profiles the QuickMLP fused networks against PyTorch
 and, if available, tiny-cuda-nn.
"""

import os
import numpy as np
import torch
import torch.nn.functional
import torch.autograd
from torchtyping import TensorType
import json
from typing import NamedTuple
import matplotlib.pyplot as plt
import time

from qmlp.import_library import load_library
from qmlp.network import FusedNetwork, SimulatedNetwork, copy_torch_to_fused
load_library()


class NetworkConfig(NamedTuple):
    """
    The largest set of configs supported by QuickMLP and tiny-cuda-nn.
    Tiny-cuda-nn only supports networks with the same number of neurons
    and activation in every layer.
    """
    n_input_dims: int
    n_output_dims: int
    activation: str
    output_activation: str
    n_neurons: int
    n_hidden_layers: int


def _create_fused_config(cfg: NetworkConfig, compile_options: dict=None) -> str:
    if compile_options is None:
        compile_options = dict()
    def convert_activation(name: str):
        if name == "None": return "identity"
        return name.lower()
    layers = []
    for _ in range(cfg.n_hidden_layers):
        layers.append({
            "n_out": cfg.n_neurons,
            "bias": False,
            "activation": convert_activation(cfg.activation)
        })
    layers.append({
        "n_out": cfg.n_output_dims,
        "bias": False,
        "activation": convert_activation(cfg.output_activation)
    })
    cfg2 = {
        "num_inputs": cfg.n_input_dims,
        "num_outputs": cfg.n_output_dims,
        "activation_specification": [
            "qmlp/builtin-activations.json"
        ],
        "encodings": [
            {
                "id": "Identity",
                "start_in": 0,
                "n_in": cfg.n_input_dims
            }
        ],
        "network": layers,
        "options": compile_options
    }
    return json.dumps(cfg2)


def create_quickmlp_network(cfg: NetworkConfig, compile_options=None) -> torch.nn.Module:
    return FusedNetwork(_create_fused_config(cfg, compile_options))


def create_pytorch16_network(cfg: NetworkConfig) -> torch.nn.Module:
    return SimulatedNetwork(_create_fused_config(cfg), torch.float16)


def create_pytorch32_network(cfg: NetworkConfig) -> torch.nn.Module:
    return SimulatedNetwork(_create_fused_config(cfg), torch.float32)


try:
    import tinycudann as tcnn

    # For now, only plain network
    def create_tinycudann_network(cfg: NetworkConfig) -> torch.nn.Module:
        cfg2 = {
            "otype": "FullyFusedMLP",
            "activation": cfg.activation,
            "output_activation": cfg.output_activation,
            "n_neurons": cfg.n_neurons,
            "n_hidden_layers": cfg.n_hidden_layers
        }
        network = tcnn.Network(cfg.n_input_dims, cfg.n_output_dims, cfg2)
        return network

    tinycudann_available = True

except ImportError:
    print("Tiny-cuda-nn not installed in the current Python environment, can't compare against it")
    tinycudann_available = False


def profile_networks():
    device = torch.device("cuda")

    n_hidden_x = [2, 4]
    n_neurons_x = [16, 32, 64]
    activation = "ReLU"
    last_activation = "None"
    n_output = 4  # RGBA

    N = 1000 * 1000
    title = "Performance for 1 mio. elements. Network: <hidden channels> x <num layers>"
    trials = 10
    startup = 2

    classes = ["PyTorch-32", "PyTorch-16", "QuickMLP-serial", "QuickMLP-parallel", "QuickMLP-parallel-skew"]
    if tinycudann_available:
        classes.append("tiny-cuda-nn")

    labels = []
    values = [list() for _ in range(len(classes))]  # ms

    for n_hidden in n_hidden_x:
        for n_neurons in n_neurons_x:
            labels.append(f"{n_neurons} x {n_hidden}")
            print(labels[-1])
            cfg = NetworkConfig(
                n_neurons,
                n_output,
                activation,
                last_activation,
                n_neurons,
                n_hidden
            )

            networks = [
                create_pytorch32_network(cfg),
                create_pytorch16_network(cfg),
                create_quickmlp_network(cfg, {
                    "parallel_weight_update": False,
                    "skew_shared_memory": False}),
                create_quickmlp_network(cfg, {
                    "parallel_weight_update": True,
                    "skew_shared_memory": False}),
                create_quickmlp_network(cfg, {
                    "parallel_weight_update": True,
                    "skew_shared_memory": True}),
            ]
            if tinycudann_available:
                networks.append(create_tinycudann_network(cfg))

            input = torch.randn((N, n_neurons), device=device, dtype=torch.float32)
            input.requires_grad_(True)
            grad_output = torch.randn((N, n_output), device=device, dtype=torch.float32)

            for i,n in enumerate(networks):
                n.to(device=device)
                if tinycudann_available:
                    tcnn.free_temporary_memory()
                torch.cuda.empty_cache()
                # Forward only
                with torch.no_grad():
                    trials_ns = []
                    for _ in range(trials):
                        input.grad = None
                        torch.cuda.synchronize(device)
                        start = time.perf_counter_ns()
                        output = n(input)
                        torch.cuda.synchronize(device)
                        end = time.perf_counter_ns()
                        trials_ns.append(end-start)
                    elapsed_inference = np.mean(trials_ns[startup:]) / (1000*1000)
                # Forward + backward
                for _ in range(trials):
                    input.grad = None
                    torch.cuda.synchronize(device)
                    start = time.perf_counter_ns()
                    output = n(input)
                    torch.autograd.backward(output, grad_output)
                    torch.cuda.synchronize(device)
                    end = time.perf_counter_ns()
                    trials_ns.append(end - start)
                elapsed_optimize = np.mean(trials_ns[startup:]) / (1000 * 1000)
                values[i].append((elapsed_inference, elapsed_optimize))
                print("  " + classes[i], elapsed_inference, elapsed_optimize)

    print(values)

    # Plot
    width_full = 0.8
    width_part = width_full / len(classes)
    x_offsets = np.linspace((-width_full+width_part)/2, (width_full-width_part)/2,
                            len(classes), endpoint=True)
    x = np.arange(len(labels))
    fig, axes = plt.subplots(ncols=2, figsize=(20, 7))
    for j,label in enumerate(["Inference", "Forward+Backward"]):
        ax: plt.Axes = axes[j]
        ax.set_xlabel(label)
        ax.set_ylabel("Time (ms)")
        for i in range(len(classes)):
            v = [values[i][k][j] for k in range(len(labels))]
            rects = ax.bar(x + x_offsets[i], v, width_part, label=classes[i])
            ax.bar_label(rects, padding=3, fmt="%.1f")
        ax.set_xticks(x, labels)
        if j==0:
            ax.legend(loc='upper left')

    fig.suptitle(title)
    fig.tight_layout()

    os.makedirs("../output", exist_ok=True)
    output_path = os.path.join("..", "output", "Performance-Networks.png")
    fig.savefig(output_path, bbox_inches="tight")
    print("Saved to", output_path)
