
if __name__ == '__main__':
    import torch
    torch.ops.load_library("bin/qmlp-pytorch-bindings.dll")
    print(dir(torch.ops.qmlp))
    print(torch.ops.qmlp.dummy)
    print(torch.ops.qmlp.dummy(4))