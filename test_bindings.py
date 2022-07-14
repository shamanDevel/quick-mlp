import torch
import torch.autograd
torch.classes.load_library("bin/qmlp.so")
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

if __name__ == '__main__':
    
    activ = FusedActivation("""{
        "id": "sineDummy",
        "forward": "z = hsin(x)",
        "adjoint": "adjx = __hmul(hcos(x), adjz)"
    }""")
    print(activ)
    
    input = torch.randn((5, 3), dtype=torch.half, device=torch.device("cuda"), requires_grad=True)
    output = activ(input)
    
    print("Input:"); print(input)
    print("Output:"); print(output)
    
    loss = torch.sum(output)
    loss.backward()
    print("Gradient:")
    print(input.grad)
    
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
    
    
    