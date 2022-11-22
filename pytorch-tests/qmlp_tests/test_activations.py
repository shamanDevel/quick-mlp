import torch
import torch.nn.functional
import torch.autograd

from qmlp.import_library import load_library
load_library()

class FusedActivation(torch.nn.Module):
    
    def __init__(self, cfg:str):
        super().__init__()
        self._cfg = cfg
        self._activation = torch.classes.qmlp.Activation(cfg)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # no grad version:
        #return self._activation.inference(input)
        
        # with autograd support
        return self._activation.forward(input)
        
    def __repr__(self):
        return f"FusedActivation(\"{self._cfg}\")"
    def __str__(self):
        return f"FusedActivation(\"{self._cfg}\")"

def _validate_activation(code:str, baseline: torch.nn.Module):
    
    activ = FusedActivation(code)
    print(activ)
    baseline.to(dtype=torch.half, device=torch.device("cuda"))

    # REFERENCE AGAINST BASELINE

    input = torch.randn((5, 3), dtype=torch.half, device=torch.device("cuda"), requires_grad=True)
    output_actual = activ(input)
    output_expected = baseline(input)

    print("Input:"); print(input)
    print("Output:"); print(output_actual)
    assert torch.allclose(output_actual, output_expected)

    output_random = torch.randn_like(output_actual)
    loss = torch.nn.functional.mse_loss(output_actual, output_random)
    loss.backward()
    grad_actual = input.grad.detach().clone()

    input.grad = None
    output_expected = baseline(input)
    loss = torch.nn.functional.mse_loss(output_expected, output_random)
    loss.backward()
    grad_expected = input.grad.detach().clone()

    #print("Gradient:")
    #print("actual:"); print(grad_actual)
    #print("expected:"); print(grad_expected)
    assert torch.allclose(grad_actual, grad_expected)

    # TORCH JIT TEST

    print("Trace:")
    activ.eval()
    traced_activ = torch.jit.trace(activ, (input, ))
    print(traced_activ.code)
    #print("Freeze:")
    traced_activ = torch.jit.freeze(traced_activ)
    print(traced_activ.code)
    #print("Save")
    torch.jit.save(traced_activ, 'test.pt')
    
    #print("Load")
    loaded_activ = torch.jit.load('test.pt')
    print(loaded_activ)
    output2 = loaded_activ(input)
    #print("Output:"); print(output2)
    assert torch.allclose(output_actual, output2)
    
def test_relu():
    _validate_activation("relu", torch.nn.ReLU())

def test_sine():
    class Sine(torch.nn.Module):
        def forward(self, x):
            return torch.sin(x)
    _validate_activation("sine", Sine())
    