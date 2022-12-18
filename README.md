# QuickMLP: Fused Networks for Scene Representation Networks

This project contains fused CUDA kernels for evaluating scene representation networks using custom activation functions, various input encodings and flexible layer specifications.

It is the successor of fV-SRN (https://github.com/shamanDevel/fV-SRN), my previous project; and adapts further idea from tiny-cuda-nn (https://github.com/NVlabs/tiny-cuda-nn), a concurrent development of such fused MLPs.

**Features**

| QuickMLP (this project)                                      | fV-SRN                                                | tiny-cuda-nn                                                 |
| ------------------------------------------------------------ | ----------------------------------------------------- | ------------------------------------------------------------ |
| full CUDA code generation and compilation during runtime     | partial CUDA runtime compilation                      | static CUDA compilation                                      |
| different sizes at every layer supported                     | limited layer sizes, all must be identical            | flexible layer sizes, but all must be identical and limited to a pre-defined set |
| supports bias in the fully-connected layers                  | supports bias                                         | no bias supported                                            |
| fully-customizable and user-definable activation functions, can be different at every layer | fixed set of activation functions                     | fixed set of activation functions                            |
| supports fused training+inference                            | only inference is fused                               | supports fused training+inference                            |
| single kernel for input encoding + network evaluation        | single kernel for input encoding + network evaluation | one kernel per input encoding, one kernel for the network    |

In other words, this project builds upon the runtime compilation idea of fV-SRN to provide fully fused kernels with maximal flexibility. The user can freely customize the network layer sizes and activation functions and arbitrarily plug them together. To support training as well, we adapt ideas of tiny-cuda-nn how the weight update and backpropagation can be realized.

**Example network specification**

TBA

**Performance**

TBA

## Compilation + Project Structure

#### Requirements

- CUDA 11.x (tested with 11.6)
- C++17-compatible compiler (tested with MSVC2019 and GCC 9.4.0)
- PyTorch for the bindings, tested with version 1.11, but should work with newer as well

Don't forget to clone this repository with submodules (`git clone --recurse-submodules`). Or if you forgot to clone with submodules, you can initialize them afterwards with `git submodule init & git submodule update`.

#### C++ library

The C++ library is located in `src_cpp/include` and `src_cpp/src`.

To compile, include the root folder of QuickMLP as a subdirectory in CMake.
Then, link against `qmlp::qmlp-library`.

#### Python / PyTorch bindings

Compile the PyTorch extension: Use `setup.py`!

1. Activate your python environment, if desired (virtualenv or conda)
2. Go to the root directory of QuickMLP.
3. Call `pip -e .`
4. Enjoy!

Note: right now, compilation is only possible in developer-mode, 
i.e. the files in this folder are directly used and not copied to the python installation.
I haven't figured out yet how to copy the resource files (kernel sources) to the installation
target in setup.py. Ideas, issues, PRs are welcome!


## API Documentation

The following documentation is written for the Python bindings, but it holds true 
for the C++ library as well. Just change the class names from `snake_case` to `CamelCase` 
and you'll have the associated C++ class / method. 

TODO: json documentation for encoding+network



## ROADMAP

Encodings:

 - [x] Identity
 - [x] 1D-6D Dense & Hash Grid
 - [x] Line Integration
 - [ ] Spherical Harmonics
 - [ ] Fourier Features

Activations:
 - [x] ReLU, CeLU, Sine, Identity
 - [ ] Snake and other trigonometric ones
 - [ ] sigmoid, ...

Network
 - [x] Fused forward evaluation
 - [x] Input Encoding + Network fusion
 - [x] Proper padding if input/output channels are not a multiple of 16
 - [x] Proper handling if the batch size is not a multiple of 16
 - [x] Gradients for the input
 - [x] Gradients for the weight matrices
 - [ ] Gradients for the bias vector


## License
QuickMLP is shipped under the permissive [MIT](https://choosealicense.com/licenses/mit/) license.

## Bug reports
If you find bugs in the library, feel free to open an issue. I will continue to use this library in future projects and therefore continue to improve and extend this library. Of course, pull requests are more than welcome.
