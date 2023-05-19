import { LLM } from "llama-node";
import { LLamaCpp, type LoadConfig } from "llama-node/dist/llm/llama-cpp.js";
import path from "path";

const model = path.resolve(process.cwd(), "../ggml-vic7b-q5_1.bin");

const llama = new LLM(LLamaCpp);

const config: LoadConfig = {
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
    nGpuLayers: 0,
};

const content = "how are you?";

const run = async () => {
    await llama.load(config);

    await llama.tokenize(content).then(console.log);
};

run();
