---
sidebar_position: 5
---

# CUDA support

llama-node supports cuda with llama.cpp backend. However, in order to use cublas with llama.cpp backend, you are supposed to do manual compilation with nvcc/gcc/clang/cmake.

Currently only Linux CUDA is supported, we seek your help to enable this on Windows.

## Prepare environment

-   Clone the project

-   Follow the [Contribution Guide](https://llama-node.vercel.app/contribution) and install all the tools/dependencies on your platform

-   Install [CUDA Tookit](https://developer.nvidia.com/cuda-downloads)

## Compile

Use cuda-compile.mts

```bash
cd packages/llama-cpp
pnpm build:cuda
```

This script also checks all the progress of CUDA compiling and gives enough hints for your compile environments

## Dynamic linking

Static linking has not been proven to work with napi-rs and llama-node, we use dynamic linking instead.

After you have done compilation, you should find the dynamic linking library under your `$HOME/.llama-node` with name of `libllama.so`.

Please add `$HOME/.llama-node` to your `LD_LIBRARY_PATH` so that the binary program can find your dynamic library.

You can run this

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.llama-node
```

or add the above line in your .bashrc or .zshrc
