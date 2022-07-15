import os
import torch
import torch.autograd
torch.classes.load_library(os.path.join(os.path.split(__file__)[0], "../bin/qmlp.so"))
print(torch.classes.loaded_libraries)

class FusedEncoding(torch.nn.Module):

    def __init__(self, cfg: str):
        super().__init__()
        self._cfg = cfg
        self._encoding = torch.classes.qmlp.Encoding(cfg)


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # no grad version:
        # return self._activation.inference(input)

        # with autograd support
        return self._encoding.forward(input)

    def __repr__(self):
        return f'FusedEncoding("""{self._cfg}""")'

    def __str__(self):
        return f'FusedEncoding("""{self._cfg}""")'


def _validate_encoding(code: str, baseline: torch.nn.Module, channels_in: int, channels_out: int):
    enc = FusedEncoding(code)
    print(enc)
    baseline.to(dtype=torch.half, device=torch.device("cuda"))
    N = 6

    # REFERENCE AGAINST BASELINE

    input = torch.randn((6, channels_in), dtype=torch.float, device=torch.device("cuda"), requires_grad=True)
    output_actual = enc(input)
    output_expected = baseline(input)

    print("Input:")
    print(input)
    print("Output:")
    print(output_actual)
    assert torch.allclose(output_actual, output_expected)

    loss = torch.sum(output_actual)
    loss.backward()
    print("Gradient:")
    print(input.grad)

    # GRAD TEST
    def double_wrap(x, fun=enc):
        return fun(x.to(dtype=torch.half)).to(dtype=torch.double)

    input_double = torch.randn((6, channels_in), dtype=torch.double, device=torch.device("cuda"), requires_grad=True)
    torch.autograd.gradcheck(double_wrap, input_double, eps=1e-1, atol=1e-1)

    # TORCH JIT TEST

    print("Trace:")
    enc.eval()
    traced_activ = torch.jit.trace(enc, (input,))
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
    print("Output:")
    print(output2)


def test_identity():
    cfg = """
{
"id": "identity",
"start_in": 0,
"n_in": 5
}
    """
    class Identity(torch.nn.Module):
        def forward(self, x):
            return x

    _validate_encoding(cfg, Identity(), 5, 5)