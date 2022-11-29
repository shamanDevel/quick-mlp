import os
import torch

_library_loaded = False


def load_library():
    if load_library.library_loaded:
        return
    if os.name == 'nt':
        _suffix = '.dll'
    else:
        _suffix = '.so'

    torch.classes.load_library(os.path.join(os.path.split(__file__)[0], "../../bin/qmlp"+_suffix))
    print(torch.classes.loaded_libraries)

    torch.classes.qmlp.QuickMLP.set_compile_debug_mode(False)
    torch.classes.qmlp.QuickMLP.set_verbose_logging(False)
    load_library.library_loaded = True

load_library.library_loaded = False
