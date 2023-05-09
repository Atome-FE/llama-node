import { LLM } from "llama-node";
import { LLamaCpp } from "llama-node/dist/llm/llama-cpp.js";
import path from "path";
const model = path.resolve(process.cwd(), "../ggml-vicuna-7b-1.1-q4_1.bin");
const llama = new LLM(LLamaCpp);
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
const template = `How are you?`;
const prompt = `A chat between a user and an assistant.
USER: ${template}
ASSISTANT:`;
const params = {
    nThreads: 4,
    nTokPredict: 2048,
    topK: 40,
    topP: 0.1,
    temp: 0.2,
    repeatPenalty: 1,
    prompt,
};
const run = async () => {
    await llama.load(config);
    await llama.createCompletion(params, (response) => {
        process.stdout.write(response.token);
    });
};
run();
