# llama-node (WIP)

WIP, not production ready, the API for nodejs may change in the future, use it with caution.

Support llama 7B model with both gglm llama and gglm alpaca, backed by [llama-rs](https://github.com/setzer22/llama-rs/tree/main/llama-rs) and [napi-rs](https://github.com/napi-rs/napi-rs)

currently supported platforms: darwin-x64, darwin-arm64, linux-x64-gnu

For testing, download model from [here](https://huggingface.co/hlhr202/alpaca-7B-ggml-int4/blob/main/ggml-alpaca-7b-q4.bin)

## Install
```bash
npm install llama-node
```

## Usage

The current version supports one inferencing session on one LLama instance in the same time,

If you want to have multiple inferencing sessions, you have to create multiple LLama instances.

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
