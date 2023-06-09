<div align="center">

# LLaMA Node

llama-node: Node.js Library for Large Language Model

[<img src="https://img.shields.io/github/actions/workflow/status/hlhr202/llama-node/llama-build.yml">](https://github.com/Atome-FE/llama-node/actions)
[<img src="https://img.shields.io/npm/l/llama-node" alt="NPM License">](https://www.npmjs.com/package/llama-node)
[<img alt="npm" src="https://img.shields.io/npm/v/llama-node">](https://www.npmjs.com/package/llama-node)
[<img alt="npm" src="https://img.shields.io/npm/types/llama-node">](https://www.npmjs.com/package/llama-node)
[<img alt="Discord" src="https://img.shields.io/discord/1106423821700960286">](https://discord.gg/dKFeCwfsDk)
[<img alt="twitter" src="https://img.shields.io/twitter/url?url=https%3A%2F%2Ftwitter.com%2Fhlhr202">](https://twitter.com/hlhr202)

  <h3><a href="https://llama-node.vercel.app/">Official Documentations</a></h3>

  <img src="./doc/assets/llama.png" width="300px" height="300px" alt="LLaMA generated by Stable diffusion"/>

<sub>Picture generated by stable diffusion.</sub>

</div>

---

- [LLaMA Node](#llama-node)
  - [Introduction](#introduction)
    - [Supported models](#supported-models)
    - [Supported platforms](#supported-platforms)
  - [Installation](#installation)
  - [Manual compilation](#manual-compilation)
  - [CUDA support](#cuda-support)
  - [Acknowledgments](#acknowledgments)
    - [Models/Inferencing tools dependencies](#modelsinferencing-tools-dependencies)
    - [Some source code comes from](#some-source-code-comes-from)
  - [Community](#community)

---

## Introduction

This project is in an early stage and is not production ready, we do not follow the semantic versioning. The API for nodejs may change in the future, use it with caution.

This is a nodejs library for inferencing llama, rwkv or llama derived models. It was built on top of [llm (originally llama-rs)](https://github.com/rustformers/llm), [llama.cpp](https://github.com/ggerganov/llama.cpp) and [rwkv.cpp](https://github.com/saharNooby/rwkv.cpp). It uses [napi-rs](https://github.com/napi-rs/napi-rs) for channel messages between node.js and llama thread.

### Supported models

llama.cpp backend supported models (in [GGML](https://github.com/ggerganov/ggml) format):

-   LLaMA 🦙
-   [Alpaca](https://github.com/ggerganov/llama.cpp#instruction-mode-with-alpaca)
-   [GPT4All](https://github.com/ggerganov/llama.cpp#using-gpt4all)
-   [Chinese LLaMA / Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
-   [Vigogne (French)](https://github.com/bofenghuang/vigogne)
-   [Vicuna](https://github.com/ggerganov/llama.cpp/discussions/643#discussioncomment-5533894)
-   [Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/)
-   [OpenBuddy 🐶 (Multilingual)](https://github.com/OpenBuddy/OpenBuddy)
-   [Pygmalion 7B / Metharme 7B](#using-pygmalion-7b--metharme-7b)

llm(llama-rs) backend supported models (in [GGML](https://github.com/ggerganov/ggml) format):

-   [GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt2)
-   [GPT-J](https://huggingface.co/docs/transformers/model_doc/gptj)
-   [LLaMA](https://huggingface.co/docs/transformers/model_doc/llama): LLaMA,
    Alpaca, Vicuna, Koala, GPT4All v1, GPT4-X, Wizard
-   [GPT-NeoX](https://huggingface.co/docs/transformers/model_doc/gpt_neox):
    GPT-NeoX, StableLM, RedPajama, Dolly v2
-   [BLOOM](https://huggingface.co/docs/transformers/model_doc/bloom): BLOOMZ

rwkv.cpp backend supported models (in [GGML](https://github.com/ggerganov/ggml) format):

-   [RWKV](https://github.com/BlinkDL/RWKV-LM)

### Supported platforms

-   darwin-x64
-   darwin-arm64
-   linux-x64-gnu (glibc >= 2.31)
-   linux-x64-musl
-   win32-x64-msvc

Node.js version: >= 16

---

## Installation

-   Install llama-node npm package

```bash
npm install llama-node
```

-   Install anyone of the inference backends (at least one)

    -   llama.cpp

    ```bash
    npm install @llama-node/llama-cpp
    ```

    -   or llm

    ```bash
    npm install @llama-node/core
    ```

    -   or rwkv.cpp

    ```bash
    npm install @llama-node/rwkv-cpp
    ```

---

## Manual compilation

Please see how to start with manual compilation on our [contribution guide](https://llama-node.vercel.app/contribution)

---

## CUDA support

Please read the document on our site to get started with [manual compilation](https://llama-node.vercel.app/docs/cuda) related to CUDA support

---

## Acknowledgments

This library was published under MIT/Apache-2.0 license. However, we strongly recommend you to cite our work/our dependencies work if you wish to reuse the code from this library.

### Models/Inferencing tools dependencies

-   LLaMA models: [facebookresearch/llama](https://github.com/facebookresearch/llama)
-   RWKV models: [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM)
-   llama.cpp: [ggreganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
-   llm: [rustformers/llm](https://github.com/rustformers/llm)
-   rwkv.cpp: [saharNooby/rwkv.cpp](https://github.com/saharNooby/rwkv.cpp)

### Some source code comes from

-   llama-cpp bindings: [sobelio/llm-chain](https://github.com/sobelio/llm-chain)
-   rwkv logits sampling: [KerfuffleV2/smolrsrwkv](https://github.com/KerfuffleV2/smolrsrwkv)

---

## Community

Join our Discord community now! [Click to join llama-node Discord](https://discord.gg/dKFeCwfsDk)
