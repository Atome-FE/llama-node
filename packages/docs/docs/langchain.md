---
sidebar_position: 4
---

# LangChain.js extension

We provide a LangChain.js compatible LLamaEmbeddings support since v0.0.28! We are not sure if it is accurate but it works :)

```typescript
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { LLamaEmbeddings } from "llama-node/dist/extensions/langchain.js";
import { LLama } from "llama-node";
import { LLamaCpp, LoadConfig } from "llama-node/dist/llm/llama-cpp.js";
import path from "path";

const model = path.resolve(process.cwd(), "../ggml-vic7b-q5_1.bin");

const llama = new LLama(LLamaCpp);

const config: LoadConfig = {
    path: model,
    enableLogging: true,
    nCtx: 1024,
    nParts: -1,
    seed: 0,
    f16Kv: false,
    logitsAll: false,
    vocabOnly: false,
    useMlock: false,
    embedding: true,
    useMmap: true,
};

llama.load(config);

const run = async () => {
    // Load the docs into the vector store
    const vectorStore = await MemoryVectorStore.fromTexts(
        ["Hello world", "Bye bye", "hello nice world"],
        [{ id: 2 }, { id: 1 }, { id: 3 }],
        new LLamaEmbeddings({ maxConcurrency: 1 }, llama)
    );

    // Search for the most similar document
    const resultOne = await vectorStore.similaritySearch("hello world", 1);

    console.log(resultOne);
};

run();

```