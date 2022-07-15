import torch
import torch.autograd
torch.classes.load_library("../bin/qmlp.so")
print(torch.classes.loaded_libraries)

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
    
    loss = torch.sum(output_actual)
    loss.backward()
    print("Gradient:")
    print(input.grad)

    # GRAD TEST
    def double_wrap(x, fun=activ):
        return fun(x.to(dtype=torch.half)).to(dtype=torch.double)

    input_double = torch.randn((5, 3), dtype=torch.double, device=torch.device("cuda"), requires_grad=True)
    torch.autograd.gradcheck(double_wrap, input_double, eps=1e-1, atol=1e-1)

    # TORCH JIT TEST

    print("Trace:")
    activ.eval()
    traced_activ = torch.jit.trace(activ, (input, ))
    print(traced_activ.code)
    print("Freeze:")
    traced_activ = torch.jit.freeze(traced_activ)
    print(traced_activ.code)
    print("Save")
    torch.jit.save(traced_activ, 'test.pt')
    
    print("Load")
    loaded_activ = torch.jit.load('test.pt')
    print(loaded_activ)
    output2 = loaded_activ(input)
    print("Output:"); print(output2)
    
def test_relu():
    _validate_activation("relu", torch.nn.ReLU())

def test_sine():
    class Sine(torch.nn.Module):
        def forward(self, x):
            return torch.sin(x)
    _validate_activation("sine", Sine())
    