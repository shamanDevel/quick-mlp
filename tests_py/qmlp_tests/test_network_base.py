import itertools
import os
import numpy as np
import torch
import torch.nn.functional
import torch.autograd
import torch.nn.init
import torch.optim
from torchtyping import TensorType
import json

from qmlp.import_library import load_library
from qmlp.network import FusedNetwork, SimulatedNetwork, copy_torch_to_fused
load_library()


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def _print_tensors(t1: np.ndarray, t2: np.ndarray, diff: np.ndarray,
                   _prefix='', _suffix='\n'):
    if t1.size > 1<<14:
        print("Array too big, can't print")
        return
    if len(t1.shape) == 2:
        print(_prefix, "[", sep="", end="")
        for i in range(t1.shape[0]):
            _print_tensors(t1[i], t2[i], diff[i], '  ', ',\n')
        print("]", end=_suffix)
    elif len(t1.shape) == 1:
        print(_prefix, "[", sep="", end="")
        for i in range(t1.shape[0]):
            v1 = t1[i]
            v2 = t2[i]
            d = diff[i]
            if d:
                print(f"{bcolors.WARNING} {v1:5.3f}|{v2:5.3f}{bcolors.ENDC}", end="")
            else:
                print(f" {v1:3f}|{v2:3f}", end="")
        print(" ]", end=_suffix)
    else:
        raise ValueError("Can print only 1D and 2D tensors")


def _compare_tensors(actual: torch.Tensor, expected: torch.Tensor, name: str,
                     rtol=1e-5, atol=1e-8, fail_threshold=0.05):
    actual = actual.detach().cpu().to(torch.float32).numpy()
    expected = expected.detach().cpu().to(torch.float32).numpy()
    same = np.isclose(actual, expected, rtol=rtol, atol=atol)
    diff = ~same
    if not np.all(same):
        print(name)
        _print_tensors(expected, actual, diff)
        fraction_failed = np.mean(diff*1.0)
        if fraction_failed > fail_threshold:
            raise AssertionError(f"{name} Tensors are not the same and the number of errors {fraction_failed*100:.2f}% have exceeded the threshold")
        else:
            print(f"{bcolors.WARNING}{name} Tensors are not the same, but the number of errors {fraction_failed*100:.2f}% is within the tolerance{bcolors.ENDC}")


def _compare_networks(actual: FusedNetwork, expected: SimulatedNetwork, grad: bool = False,
                      rtol=1e-5, atol=1e-8):
    for layer in range(expected.num_layers()):
        w_expected = expected.get_weight(layer, False, grad)
        w_actual = actual.get_weight(layer, False, grad)
        _compare_tensors(w_actual, w_expected,
                         f"Layer {layer} weights {'(grad)' if grad else ''}: ",
                         rtol, atol)

        if expected.has_bias(layer):
            b_expected = expected.get_weight(layer, True, grad)
            b_actual = actual.get_weight(layer, True, grad)
            _compare_tensors(b_actual, b_expected,
                             f"Bias {layer} weights {'(grad)' if grad else ''}: ",
                             rtol, atol)


def _test_network(cfg: str):
    network_fused = FusedNetwork(cfg)
    network_torch = SimulatedNetwork(cfg)
    copy_torch_to_fused(network_fused, network_torch)
    n_in = network_fused.channels_in()
    n_out = network_fused.channels_out()

    device = torch.device("cuda")
    rtol = 1e-2
    atol = 1e-3
    rtol_wgrad = 1e-1

    network_fused.to(device)
    network_torch.to(device)

    # number of elements to test
    Nx = [16, 32, 128, 513, 48621]
    for N in Nx:
        print(f"Run with N={N} input elements")
        input = torch.randn((N, n_in), device=device, dtype=torch.float32)
        grad_output = torch.randn((N, n_out), device=device, dtype=torch.float32)

        # 1. Test if the output of the inference pass matches
        with torch.no_grad():
            output_expected = network_torch(input)
            output_actual = network_fused(input)
            _compare_tensors(output_actual, output_expected,
                             f"Inference pass with {N} inputs",
                             rtol, atol)

        # backward
        for p in network_torch.parameters():
            p.grad = None
        for p in network_fused.parameters():
            p.grad = None

        # 2. Test if the output of the forward pass matches
        input_expected = torch.clone(input)
        input_actual = torch.clone(input)
        input_expected.requires_grad_(True)
        input_actual.requires_grad_(True)
        output_expected = network_torch(input_expected)
        output_actual = network_fused(input_actual)
        _compare_tensors(output_actual, output_expected,
                         f"Forward pass with {N} inputs",
                         rtol, atol)

        # 3. backprop and compare gradients
        torch.autograd.backward(output_expected, grad_output)
        torch.autograd.backward(output_actual, grad_output)

        # Compare input gradients
        with torch.no_grad():
            _compare_tensors(input_actual.grad, input_expected.grad,
                             f"Backward pass with {N} inputs, inputs",
                             rtol, atol)

        # Compare parameter gradients
        with torch.no_grad():
            _compare_networks(network_fused, network_torch,
                              True, rtol_wgrad, atol)


def test_network_single_layer():

    channels = [16, 32, 48]
    activations = ["celu", "identity"]
    for c_in, c_out, activation in itertools.product(channels, channels, activations):
        cfg = f"""{{
    "num_inputs": {c_in},
    "num_outputs": {c_out},
    "activation_specification": [
        "qmlp/builtin-activations.json"
      ],
    "encodings": [
        {{
          "id": "Identity",
          "start_in": 0,
          "n_in": {c_in}
        }}
      ],
    "network": [
        {{
          "n_out": {c_out},
          "bias": false,
          "activation": "{activation}"
        }}
      ]
    }}"""
        print(cfg)
        _test_network(cfg)


def test_network_two_layers():
    channels = [16, 32, 48]
    hidden_channels = [16, 32, 48]
    activations_hidden = ["celu", "identity"]
    activations_last = ["celu", "identity"]
    for c_in, c_out, c_hidden, activation_hidden, activation_last in itertools.product(
            channels, channels, hidden_channels, activations_hidden, activations_last):
        cfg = f"""{{
    "num_inputs": {c_in},
    "num_outputs": {c_out},
    "activation_specification": [
        "qmlp/builtin-activations.json"
      ],
    "encodings": [
        {{
          "id": "Identity",
          "start_in": 0,
          "n_in": {c_in}
        }}
      ],
    "network": [
        {{
          "n_out": {c_hidden},
          "bias": false,
          "activation": "{activation_hidden}"
        }},
        {{
          "n_out": {c_out},
          "bias": false,
          "activation": "{activation_last}"
        }}
      ]
    }}"""
        print(cfg)
        _test_network(cfg)


def test_network_three_layers():
    channels = [16, 32, 48]
    hidden_channels = [16, 32, 48]
    activations_hidden = ["celu", "identity"]
    activations_last = ["celu", "identity"]
    for c_in, c_out, c_hidden, activation_hidden, activation_last in itertools.product(
            channels, channels, hidden_channels, activations_hidden, activations_last):
        cfg = f"""{{
    "num_inputs": {c_in},
    "num_outputs": {c_out},
    "activation_specification": [
        "qmlp/builtin-activations.json"
      ],
    "encodings": [
        {{
          "id": "Identity",
          "start_in": 0,
          "n_in": {c_in}
        }}
      ],
    "network": [
        {{
          "n_out": {c_hidden},
          "bias": false,
          "activation": "{activation_hidden}"
        }},
        {{
          "n_out": {c_hidden},
          "bias": false,
          "activation": "{activation_hidden}"
        }},
        {{
          "n_out": {c_out},
          "bias": false,
          "activation": "{activation_last}"
        }}
      ]
    }}"""
        print(cfg)
        _test_network(cfg)
