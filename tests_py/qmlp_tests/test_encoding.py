import torch
import torch.nn.functional
import torch.autograd
import torch.nn.init
from typing import Optional

from qmlp.encoding import FusedEncoding


def _validate_encoding(code: str, baseline: Optional[torch.nn.Module], channels_in: int):
    enc = FusedEncoding(code)
    print(enc)
    print("Num outputs:", enc.num_output_channels())
    baseline.to(dtype=torch.half, device=torch.device("cuda"))
    N = 8

    if enc.has_parameters():
        parameters = enc.create_parameter_tensor()
        print("Encoding requires parameters, shape:", parameters.shape, ", dtype:", parameters.dtype)
    else:
        parameters = None

    # REFERENCE AGAINST BASELINE

    input = torch.rand((N, channels_in), dtype=torch.float, device=torch.device("cuda")) * 1.8 - 0.9
    #if channels_in==2:
    #    input[:4,:] = torch.tensor([
    #        [-1,-1], [-1,1], [1,-1], [1,1]
    #    ], dtype=torch.float, device=torch.device("cuda"))
    input = input.detach().requires_grad_(True)
    if parameters is not None: parameters.grad = None
    output_actual = enc(input, parameters)
    assert output_actual.shape[0] == N
    assert output_actual.shape[1] == enc.num_output_channels()

    print("Input:"); print(input)
    print("Output:"); print(output_actual)

    output_random = torch.randn_like(output_actual)
    loss = torch.nn.functional.mse_loss(output_actual, output_random)
    loss.backward()
    grad_input_actual = input.grad.detach().clone()
    if enc.has_parameters():
        grad_parameter_actual = parameters.grad.detach().clone()

    if baseline is not None:
        input.grad = None
        if parameters is not None: parameters.grad = None
        output_expected = baseline(input, parameters)
        print("Expected:"); print(output_expected)
        assert torch.allclose(output_actual, output_expected)
        loss = torch.nn.functional.mse_loss(output_expected, output_random)
        loss.backward()
        grad_input_expected = input.grad.detach().clone()
        if enc.has_parameters():
            grad_parameter_expected = parameters.grad.detach().clone()

    print("Gradient for the inputs:")
    print("actual:"); print(grad_input_actual)
    if baseline is not None:
        print("expected:"); print(grad_input_expected)
        assert torch.allclose(grad_input_actual, grad_input_expected)
    if enc.has_parameters():
        print("Gradient for the parameters:")
        print("actual:"); print(grad_parameter_actual)
        if baseline is not None:
            print("expected:"); print(grad_parameter_expected)
            assert torch.allclose(grad_parameter_actual, grad_parameter_expected)

    # GRAD TEST
    def wrap_input(x, params=parameters, fun=enc):
        return fun(x.to(dtype=torch.float), params.detach() if params is not None else None).to(dtype=torch.double)
    def wrap_params(p, i=input, fun=enc):
        return fun(i.detach(), p.to(dtype=torch.float)).to(dtype=torch.double)

    input_double = input.detach().to(torch.double).requires_grad_(True)
    param_double = torch.randn(parameters.shape, dtype=torch.double, device=torch.device("cuda"), requires_grad=True) if parameters is not None else None
    torch.autograd.gradcheck(wrap_input, input_double, eps=1e-1, atol=1e-1)
    if enc.has_parameters():
        torch.autograd.gradcheck(wrap_params, param_double, eps=1e-1, atol=1e-1)

    # TORCH JIT TEST

    print("Trace:")
    enc.eval()
    if enc.has_parameters():
        traced_activ = torch.jit.trace(enc, (input,parameters))
    else:
        traced_activ = torch.jit.trace(enc, (input,))
    print(traced_activ.code)
    #print("Freeze:")
    traced_activ = torch.jit.freeze(traced_activ)
    print(traced_activ.code)
    #print("Save")
    torch.jit.save(traced_activ, 'test.pt')

    #print("Load")
    loaded_activ = torch.jit.load('test.pt')
    #print(loaded_activ)
    if enc.has_parameters():
        output2 = loaded_activ(input, parameters)
    else:
        output2 = loaded_activ(input)
    #print("Output:")
    #print(output2)
    assert torch.allclose(output_actual, output2)

def test_identity():
    cfg = """
{
"id": "Identity",
"start_in": 0,
"n_in": 5
}
    """
    class Identity(torch.nn.Module):
        def forward(self, x, parameters_unused):
            return x

    _validate_encoding(cfg, Identity(), 5)


def test_densegrid_1d():
    resolution = 2
    channels = 4

    cfg = f"""
{{
"id": "HashGrid",
"start_in": 0,
"n_in": 1,
"n_levels": 1,
"n_features_per_level": {channels},
"log2_hashmap_size": -1,
"min_resolution": {resolution},
"max_resolution": 16,
"combination_mode": "add",
"bounding_box_min": [-1],
"bounding_box_size": [2]
}}
    """

    class Densegrid1D(torch.nn.Module):
        def __init__(self, r, c):
            super().__init__()
            self._r = r
            self._c = c
        def forward(self, x, parameters):
            input = parameters.reshape(1, self._r, 1, self._c).permute((0,3,2,1))
            grid = x.reshape(1, -1, 1, 1)
            grid = torch.cat([grid, grid], dim=3)
            output = torch.nn.functional.grid_sample(input, grid, mode='bilinear', align_corners=True, padding_mode='border')
            return output[0,:,:,0].t()

    _validate_encoding(cfg, Densegrid1D(resolution, channels), 1)

def test_densegrid_2d():
    resolution = 2
    channels = 4

    cfg = f"""
{{
"id": "HashGrid",
"start_in": 0,
"n_in": 2,
"n_levels": 1,
"n_features_per_level": {channels},
"log2_hashmap_size": -1,
"min_resolution": {resolution},
"max_resolution": 16,
"combination_mode": "add",
"bounding_box_min": [-1,-1],
"bounding_box_size": [2,2]
}}
    """

    class Densegrid2D(torch.nn.Module):
        def __init__(self, r, c):
            super().__init__()
            self._r = r
            self._c = c
        def forward(self, x, parameters):
            input = parameters.reshape(1, self._r, self._r, self._c).permute((0,3,2,1))
            grid = x.reshape(1, -1, 1, 2)
            output = torch.nn.functional.grid_sample(input, grid, mode='bilinear', align_corners=True, padding_mode='border')
            return output[0,:,:,0].t()

    _validate_encoding(cfg, Densegrid2D(resolution, channels), 2)

