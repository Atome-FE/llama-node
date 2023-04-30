---
title: Contribution guide
---

# Contribution guide

---

## Prepare environments

These are the build tools you need to install in your local dev environment

-   [Node.js](https://nodejs.org/) >= 16, for running Node.js

-   [Rust](https://www.rust-lang.org/tools/install) follow rustup for installation, for building rust code

-   [Pnpm](https://pnpm.io/), a Node.js package manager that llama-node used. We only define Node.js workspaces in pnpm format.

-   [CMake](https://cmake.org/), for building c++ backends.

### For Linux

Here we use Ubuntu as an example. For users of other Linux distributions, please install similar tools for your respective distribution.

-   [GCC](https://gcc.gnu.org/), for compiling C/C++ code

    Ubuntu/Debian user can use `apt install build-essential`

-   [Musl tool](https://musl.cc/), for cross compile musl libc on GCC

    Ubuntu/Debian user can use `apt install musl-tools`

### For Mac

-   [XCode](https://developer.apple.com/xcode/), for compiling C/C++ code

### For Windows

-   [Visual Studio with C/C++ Component](https://visualstudio.microsoft.com/vs/features/cplusplus/), for compiling C/C++ code

---

## Cross compilation

Cross compilation has a lot of tricks and magics. You may encounter issues that cannot be found in this documentation or even through online research. If you cannot handle it properly, please contact llama-node maintainers for help.

We added cross compilation on OSX environment, which enable MacOS users to compile both x64 and aarch64(ARM) programs.

We also provide cross compilation on Linux environment, but that one only compiles for both x64-gcc and x64-musl which served for different libc containers.

### For Mac

Please add rust target aarch64 on x64 system, or x64 target on aarch64 system.

```shell
# run this on x86-64 MacOS
rustup target add aarch64-apple-darwin
```

```shell
# run this on aarch64 MacOS
rustup target add x86_64-apple-darwin
```

### For Linux

Here we use x86-64 Ubuntu (which provides Glibc/GCC as default standard C compilation) as an example. For users of other Linux distributions, please install targets for your respective distribution.

```shell
rustup target add x86_64-unknown-linux-musl
```

---

## Prepare codebase

-   Initialize submodules

    ```shell
    git submodule update --init --recursive
    ```

-   Install Node.js dependencies

    Here we use --ignore-scripts flags for the first time installation to avoid some preinstall scripts error

    ```shell
    pnpm install --ignore-scripts
    ```

---

## Build backends

-   Build for llama-cpp

    ```shell
    pnpm build:llama-cpp
    ```

-   Build for llama-rs

    ```shell
    pnpm build:llama-rs
    ```

-   Build for rwkv-cpp

    ```shell
    pnpm build:rwkv-cpp
    ```

---

## Build Typescript wrapper

```shell
pnpm build
```