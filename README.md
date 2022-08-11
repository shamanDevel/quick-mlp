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

The project is split into two main parts:
First, the library in the `include/` and `src` folders contain the algorithms and uses an abstraction of tensors that is completely independent of deep learning libraries. This library can be integrated into other C++ projects for real-time training or inference.
Second, the PyTorch bindings in `pytorch-bindings` contains interface code to use the fused kernels in regular PyTorch Python scripts.

#### Requirements

- CUDA 11.x (tested with 11.6)
- C++17-compatible compiler (tested with MSVC2019 and GCC 9.4.0)
- PyTorch, tested with version 1.11, but should work with newer as well

#### Compilation

Compiling the stand-along library (no PyTorch) for C++-projects: Use CMake!
Include this repository as a subdirectory and link your library against the target `qmlp::qmlp-library`.

Compile the PyTorch extension: Use `setup.py`!
In the root directory of the project, call 

```bash
python setup.py build
mkdir -p bin
cp build/lib.linux-x86_64-3.9/qmlp.so bin/
```
Note: The path in the last line is specific for Python 3.9 on a Linux machine. For other OS and other Python version, the compiled file will be placed somewhere else. See the log of setup.py for the details.

After compilation, the custom kernels can be imported into PyTorch via

```python
import torch # Run in Python
torch.classes.load_library('bin/qmlp.so') # path to the library
```



## Python-API Documentation

TBA, see the test scripts in `pytorch-tests` for now.



## License
QuickMLP is shipped under the permissive [MIT](https://choosealicense.com/licenses/mit/) license.

## Bug reports
If you find bugs in the library, feel free to open an issue. I will continue to use this library in future projects and therefore continue to improve and extend this library. Of course, pull requests are more than welcome.