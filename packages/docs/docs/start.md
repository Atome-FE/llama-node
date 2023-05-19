---
sidebar_position: 1
---

# Get Started

Large Language Model on node.js.

This project is in an early stage, the API for nodejs may change in the future, use it with caution.

---

## Prerequisites

- [Node.js](https://nodejs.org/en/download/) version 16 or above
  
- (Optional) [Typescript](https://www.typescriptlang.org/): When you want statically typed interfaces

- (Optional) [Python 3](https://www.python.org/downloads/): When you need to convert the pth to ggml format

- (Optional) Rust/C++ compiling toolchains: When you need self compilation
  
  - [Rust](https://www.rust-lang.org/tools/install) for building rust node api
  
  - [CMake](https://cmake.org/) for building llama.cpp project
  
  - Clang/GNU/MSVC C++ compiler for compiling native C/C++ bindings, you can choose:
    
    - [build-essential](https://packages.ubuntu.com/jammy/build-essential) for Ubuntu (run ```apt install build-essential```)
    
    - [XCode](https://developer.apple.com/xcode/) for MacOS (run ```xcode-select --install```)

    - [Visual Studio](https://visualstudio.microsoft.com/) for Windows (Install C/C++ components)

---

## Compatibility

Currently supported models (All of these should be converted to [GGML](https://github.com/ggerganov/ggml) format):
- [LLaMA](https://github.com/facebookresearch/llama)
- [RWKV](https://github.com/BlinkDL/RWKV-LM)
- [Alpaca](https://github.com/ggerganov/llama.cpp#instruction-mode-with-alpaca)
- [GPT4All](https://github.com/ggerganov/llama.cpp#using-gpt4all)
- [Chinese LLaMA / Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
- [Vigogne (French)](https://github.com/bofenghuang/vigogne)
- [Vicuna](https://github.com/ggerganov/llama.cpp/discussions/643#discussioncomment-5533894)
- [Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/)

Supported platforms:
- darwin-x64
- darwin-arm64
- linux-x64-gnu (glibc >= 2.31)
- linux-x64-musl
- win32-x64-msvc

Node.js version: >= 16

---

## Installation

- Install llama-node npm package

```bash
npm install llama-node
```

- Install anyone of the inference backends (at least one)
  
  - llama.cpp
  
  ```bash
  npm install @llama-node/llama-cpp
  ```

  - or llm-rs
  
  ```bash
  npm install @llama-node/core
  ```

  - or rwkv.cpp
  
  ```bash
  npm install @llama-node/rwkv-cpp
  ```

---

## Getting Model

- For llama and its derived models:

  The llama-node uses llm-rs/llama.cpp under the hook and uses the model format (GGML/GGMF/GGJT) derived from llama.cpp. Due to the fact that the meta-release model is only used for research purposes, this project does not provide model downloads. If you have obtained the original .pth model, please read the [document](https://github.com/ggerganov/llama.cpp#prepare-data--run) and use the conversion tool provided by llama.cpp for conversion.

- For RWKV models:
  
  RWKV is open source model developed by [PENG Bo](https://github.com/BlinkDL). All the model weights and training codes are open source. Our rwkv backend uses rwkv.cpp native bindings which also utilized the GGML tensor formats. You can download the GGML quantized model from [here](https://huggingface.co/Malan/ggml-rwkv-4-raven-Q4_1_0) or convert it by following the [document](https://github.com/saharNooby/rwkv.cpp)

---

## First example

This is your first example that uses llama.cpp as inference backend, make sure you have installed ```@llama-node/llama-cpp``` package.

```js
// index.mjs
import { LLM } from "llama-node";
import { LLamaCpp } from "llama-node/dist/llm/llama-cpp.js";
import path from "path";

const model = path.resolve(process.cwd(), "../ggml-vic7b-q5_1.bin");
const llama = new LLM(LLamaCpp);
const config = {
    modelPath: model,
    enableLogging: true,
    nCtx: 1024,
    seed: 0,
    f16Kv: false,
    logitsAll: false,
    vocabOnly: false,
    useMlock: false,
    embedding: false,
    useMmap: true,
};

const template = `How are you?`;
const prompt = `A chat between a user and an assistant.
USER: ${template}
ASSISTANT:`;

const run = async () => {
  await llama.load(config);

  await llama.createCompletion({
      nThreads: 4,
      nTokPredict: 2048,
      topK: 40,
      topP: 0.1,
      temp: 0.2,
      repeatPenalty: 1,
      prompt,
  }, (response) => {
      process.stdout.write(response.token);
  });
}

run();
```

To run this example

```bash
node index.mjs
```

## More examples

Visit our example folder [here](https://github.com/Atome-FE/llama-node/tree/main/example)

## Acknowledgments

This library was published under MIT/Apache-2.0 license. However, we stronly recommend you to cite our work/our dependencies work if you wish to reuse the code from this library.

### Models/Inferencing tools dependencies:

- LLaMA models: [facebookresearch/llama](https://github.com/facebookresearch/llama)
- RWKV models:  [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM)
- llama.cpp:    [ggreganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
- llm-rs:     [rustformers/llm](https://github.com/rustformers/llm)
- rwkv.cpp:     [saharNooby/rwkv.cpp](https://github.com/saharNooby/rwkv.cpp)

### Some source code comes from:

- cpp-rust bindings build scripts:  [sobelio/llm-chain](https://github.com/sobelio/llm-chain)
- rwkv logits sampling:             [KerfuffleV2/smolrsrwkv](https://github.com/KerfuffleV2/smolrsrwkv)