import { LLama } from "../../src";
import { LLamaCpp, LoadConfig } from "../../src/llm/llama-cpp";
import path from "path";

const model = path.resolve(process.cwd(), "./ggml-vicuna-7b-4bit-rev1.bin");

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
    embedding: false,
};

llama.load(config);

const content = "how are you?";

llama.tokenize({ content, nCtx: 2048 }).then(console.log);
