---
sidebar_position: 1
---

# 开始

Node.js运行的大语言模型LLaMA。

这个项目处于早期阶段，nodejs的API可能会在未来发生变化，请谨慎使用。

---

## 前置要求

- [Node.js](https://nodejs.org/en/download/) 16 或更高
  
- (可选) [Typescript](https://www.typescriptlang.org/)：当你需要静态类型的时候

- (可选) [Python 3](https://www.python.org/downloads/)：当你需要把pth模型转换到ggml格式的时候

- (可选) Rust/C++ 编译工具链：当你需要自行构建二进制的时候
  
  - [Rust](https://www.rust-lang.org/tools/install) 构建Rust到Nodejs API
  
  - [CMake](https://cmake.org/) 构建llama.cpp依赖项目
  
  - Clang/GNU/MSVC C++ 编译器用于编译二进制绑定，你可以根据系统选择以下工具：
    
    - [build-essential](https://packages.ubuntu.com/jammy/build-essential) for Ubuntu (run ```apt install build-essential```)
    
    - [XCode](https://developer.apple.com/xcode/) for MacOS (run ```xcode-select --install```)

    - [Visual Studio](https://visualstudio.microsoft.com/) for Windows (安装 C/C++ 组件)

---

## 跨平台

当前支持平台：

- darwin-x64
- darwin-arm64
- linux-x64-gnu (glibc >= 2.31)
- linux-x64-musl
- win32-x64-msvc

---

## 安装

- 安装 llama-node npm 包

```bash
npm install llama-node
```

- 安装任意一个推理后端 (至少一个)
  
  - llama.cpp
  
  ```bash
  npm install @llama-node/llama-cpp
  ```

  - 或者 llm-rs
  
  ```bash
  npm install @llama-node/core
  ```

---

## 获取模型

llama-node底层调用llm-rs或llama.cpp，它使用的模型格式源自llama.cpp。由于meta发布模型仅用于研究机构测试，本项目不提供模型下载。如果你获取到了 .pth 原始模型，请阅读 [该文档](https://github.com/ggerganov/llama.cpp#prepare-data--run) 并使用llama.cpp提供的convert工具进行转化。

---

## 第一个例子

以下是使用llama.cpp作为推理后端的第一个例子, 请确认你已经安装了 ```@llama-node/llama-cpp``` NPM依赖.

```js
// index.mjs
import { LLama } from "llama-node";
import { LLamaCpp } from "llama-node/dist/llm/llama-cpp.js";
import path from "path";

const model = path.resolve(process.cwd(), "../ggml-vic7b-q5_1.bin");
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

运行该例子，请在bash输入

```bash
node index.mjs
```

## 更多例子

请访问我们的Github仓库目录，在[这里](https://github.com/Atome-FE/llama-node/tree/main/example)