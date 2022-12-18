import os
import torch
import itertools

def load_library():
    if load_library.library_loaded:
        return
    if os.name == 'nt':
        search_suffix = ['.dll', ".pyd"]
    else:
        search_suffix = ['.so']

    search_paths = [
        "..",  # setup.py install
        "../../bin"  # CMake build
    ]

    for search_path, suffix in itertools.product(search_paths, search_suffix):
        path = os.path.join(os.path.split(__file__)[0], search_path, "qmlp_cu"+suffix)
        if os.path.exists(path):
            torch.classes.load_library(path)
            print("QMLP binaries loaded from", path)
            if not torch.classes.qmlp_cu.QuickMLP.is_cuda_available():
                # CUDA not found or can't be initialized!!
                # While we could still create instances of the classes in the backend,
                # we can't launch any kernels.
                # Hence, I'm going to throw an error here so that user code can respond early and
                # switch to different implementations.
                raise ImportError("CUDA not found, QuickMLP can't be imported")

            torch.classes.qmlp_cu.QuickMLP.set_compile_debug_mode(False)
            torch.classes.qmlp_cu.QuickMLP.set_log_level("info")
            load_library.library_loaded = True

            return

    raise ValueError("QMLP binaries not found. Check your installation or adapt the search paths")

load_library.library_loaded = False
