# llama-node (WIP)

WIP, not production ready, the API for nodejs may change in the future, use it with caution.

Support llama 7B model with both gglm llama and gglm alpaca, backed by llama-rs and napi-rs

currently supported platforms: darwin-x64, darwin-arm64, linux-x64-gnu

For testing, download model from [here](https://huggingface.co/hlhr202/alpaca-7B-ggml-int4/blob/main/ggml-alpaca-7b-q4.bin)

## Install
```bash
npm install @llama-node/core
```

## Usage

Due to limited threading strategy, the current version does not support multiple session

I will provide an async based API in the next big version

```typescript
import { LLama } from "@llama-node/core";
import path from "path";

const model = path.resolve(process.cwd(), "./ggml-alpaca-7b-q4.bin");

const llama = LLama.create({ path: model });

const prompt = "test";

llama.onGenerated((response) => {
    switch (response.type) {
        case "DATA": {
            process.stdout.write(response.data.token);
            if (response.data.completed) {
                llama.terminate();
            }
            break;
        }
        case "ERROR": {
            console.log(response);
            llama.terminate();
            break;
        }
    }
});

llama.inference({ prompt });
```
