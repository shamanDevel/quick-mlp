import torch
import torch.autograd
torch.classes.load_library("bin/qmlp.so")
print(torch.classes.loaded_libraries)

class _ActivationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input:torch.Tensor, activation):
        ctx.save_for_backward(input)
        ctx.activation = activation
        
        return activation.forward(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        activation = ctx.activation
        
        return activation.adjoint(input, grad_output), None
        
class FusedActivation(torch.nn.Module):
    
    def __init__(self, cfg:str):
        super().__init__()
        self._cfg = cfg
        self._activation = torch.classes.qmlp.Activation(cfg)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return _ActivationFunction.apply(input, self._activation)
        
    def __repr__(self):
        return f"FusedActivation(\"{self._cfg}\")"
    def __str__(self):
        return f"FusedActivation(\"{self._cfg}\")"

if __name__ == '__main__':
    
    activ = FusedActivation("relu")
    print(activ)
    
    input = torch.randn((5, 3), dtype=torch.half, device=torch.device("cuda"), requires_grad=True)
    output = activ(input)
    
    print("Input:"); print(input)
    print("Output:"); print(output)
    
    loss = torch.sum(output)
    loss.backward()
    print("Gradient:")
    print(input.grad)
    
    