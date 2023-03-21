# llama-node

![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/hlhr202/llama-node/llama-build.yml)
![NPM](https://img.shields.io/npm/l/llama-node)
![npm](https://img.shields.io/npm/v/llama-node)
![npm type definitions](https://img.shields.io/npm/types/llama-node)

WIP, the API for nodejs may change in the future, use it with caution.

Support llama 7B model with both gglm llama and gglm alpaca, backed by [llama-rs](https://github.com/setzer22/llama-rs/tree/main/llama-rs) and [napi-rs](https://github.com/napi-rs/napi-rs)

currently supported platforms: darwin-x64, darwin-arm64, linux-x64-gnu

Download one of the llama ggml models from the following links:
- [alpaca 7B int4](https://huggingface.co/hlhr202/alpaca-7B-ggml-int4/blob/main/ggml-alpaca-7b-q4.bin)
- [llama 7B int4](https://huggingface.co/hlhr202/llama-7B-ggml-int4/blob/main/ggml-model-q4_0.bin)


## Install
```bash
npm install llama-node
```

## Self built

Make sure you have installed rust

```bash
cd packages/core
npm run build
```


## Usage

The current version supports only one inference session on one LLama instance at the same time

If you wish to have multiple inference sessions concurrently, you need to create multiple LLama instances

```typescript
import path from "path";
import { LLamaClient } from "llama-node";

const model = path.resolve(process.cwd(), "./ggml-alpaca-7b-q4.bin");

const client = new LLamaClient({ path: model, numCtxTokens: 4096 }, true);

const prompt = `// Show an example of counter component in react.js. it has increment and decrement buttons where they change the state by 1.
export const Counter => {`;

client.createTextCompletion(
    {
        prompt,
        numPredict: BigInt(2048),
        temp: 0.2,
        topP: 0.8,
        topK: BigInt(40),
        repeatPenalty: 1,
        repeatLastN: BigInt(512),
    },
    (res) => {
        process.stdout.write(res.token);
    }
);
```

## Future plan
- [ ] prompt extensions
- [ ] more platforms and cross compile
- [ ] better github CI