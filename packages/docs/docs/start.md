---
sidebar_position: 1
---

# Get Started

Large Language Model LLaMA on node.js.

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

## Platforms

Currently supported platforms:

- darwin-x64
- darwin-arm64
- linux-x64-gnu (glibc >= 2.31)
- linux-x64-musl
- win32-x64-msvc

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

  - or llama-rs
  
  ```bash
  npm install @llama-node/core
  ```

---

## Getting Model

The llama-node uses llama-rs/llama.cpp under the hook and uses the model format (GGML/GGMF/GGJT) derived from llama.cpp. Due to the fact that the meta-release model is only used for research purposes, this project does not provide model downloads. If you have obtained the original .pth model, please read the [document](https://github.com/ggerganov/llama.cpp#prepare-data--run) and use the conversion tool provided by llama.cpp for conversion.

---

## First example

This is your first example that uses llama.cpp as inference backend, make sure you have installed ```@llama-node/llama-cpp``` package.

```js
// index.mjs
import { LLama } from "llama-node";
import { LLamaCpp } from "llama-node/dist/llm/llama-cpp.js";
import path from "path";

const model = path.resolve(process.cwd(), "../ggml-vicuna-7b-1.1-q4_1.bin");
const llama = new LLama(LLamaCpp);
const config = {
    path: model,
    enableLogging: true,
    nCtx: 1024,
    nParts: -1,
    seed: 0,
    f16Kv: false,
    logitsAll: false,
    vocabOnly: false,
    useMlock: false,
    embedding: false,
    useMmap: true,
};
llama.load(config);

const template = `How are you?`;
const prompt = `### Human:
${template}
### Assistant:`;

llama.createCompletion({
    nThreads: 4,
    nTokPredict: 2048,
    topK: 40,
    topP: 0.1,
    temp: 0.2,
    repeatPenalty: 1,
    stopSequence: "### Human",
    prompt,
}, (response) => {
    process.stdout.write(response.token);
});
```

To run this example

```bash
node index.mjs
```

## More examples

Visit our example folder [here](https://github.com/Atome-FE/llama-node/tree/main/example)