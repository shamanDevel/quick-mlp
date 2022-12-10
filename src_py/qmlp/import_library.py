import os
import torch


def load_library():
    if load_library.library_loaded:
        return
    if os.name == 'nt':
        _suffix = '.dll'
    else:
        _suffix = '.so'

    search_paths = [
        "..",  # setup.py install
        "../../bin"  # CMake build
    ]

    for search_path in search_paths:
        path = os.path.join(os.path.split(__file__)[0], search_path, "qmlp_cu"+_suffix)
        if os.path.exists(path):
            torch.classes.load_library(path)
            print("QMLP binaries loaded from", path)
            print(torch.classes.loaded_libraries)

            torch.classes.qmlp.QuickMLP.set_compile_debug_mode(False)
            torch.classes.qmlp.QuickMLP.set_verbose_logging(False)
            load_library.library_loaded = True

            return

    raise ValueError("QMLP binaries not found. Check your installation or adapt the search paths")

load_library.library_loaded = False
