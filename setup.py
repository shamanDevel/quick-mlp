from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, _get_cuda_arch_flags, _join_cuda_home
import os
import glob
import itertools
import re

_arch_flags = _get_cuda_arch_flags()
print('arch flags:', _arch_flags)

_root = os.path.split(os.path.abspath(__file__))[0]
print("root path:", _root)

def get_files(base, filter=".*"):
    fx1 = glob.iglob(base + "/**/*.cu", recursive=True)
    fx2 = glob.iglob(base + "/**/*.cpp", recursive=True)
    fx3 = glob.iglob(base + "/**/*.c", recursive=True)
    fx = itertools.chain(fx1, fx2, fx3)
    prog = re.compile(filter)
    return [os.path.relpath(f, _root) for f in fx if prog.fullmatch(f)]

_qmlp_files = get_files(os.path.join(_root, 'src'))
_binding_files = get_files(os.path.join(_root, 'pytorch-bindings'))
_thirdparty_files = get_files(os.path.join(_root, 'third-party/cuda-kernel-loader/src'))

print("qmlp files:", _qmlp_files)
print("binding files:", _binding_files)
print("third party files:", _thirdparty_files)

_include_dirs = [
    '%s/include'%_root,
    '%s/third-party/cuda-kernel-loader/include'%_root,
    '%s/third-party/json/single_include'%_root,
    '%s/third-party/tinyformat'%_root,
    '/usr/include',
]

_libraries = [
    'cuda',
    'nvrtc',
    'curand'
]

_common_args = [
    '/std:c++17' if os.name=='nt' else '-std=c++17',
    '-DCKL_NVCC_INCLUDE_DIR=%s'%_join_cuda_home('include'),
]

setup(
    name='qmlp',
    ext_modules=[
        CUDAExtension('qmlp',
            _qmlp_files+_binding_files+_thirdparty_files,
            extra_compile_args = {
                'cxx': _common_args,
                'nvcc': _common_args+["--extended-lambda"]
            },
            include_dirs = _include_dirs,
            libraries = _libraries),
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
    })
