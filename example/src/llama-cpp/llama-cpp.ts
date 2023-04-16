import { LLama } from "llama-node";
import { LLamaCpp, LoadConfig } from "llama-node/dist/llm/llama-cpp.js";
import path from "path";

const model = path.resolve(process.cwd(), "../ggml-vicuna-7b-4bit-rev1.bin");

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
    useMmap: true,
};

llama.load(config);

const template = `How are you`;

const prompt = `### Human:

${template}

### Assistant:`;

llama.createCompletion(
    {
        nThreads: 4,
        nTokPredict: 2048,
        topK: 40,
        topP: 0.1,
        temp: 0.2,
        repeatPenalty: 1,
        stopSequence: "### Human",
        prompt,
    },
    (response) => {
        process.stdout.write(response.token);
    }
);
