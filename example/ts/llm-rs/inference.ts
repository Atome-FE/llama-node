import { Generate, ModelType } from "@llama-node/core";
import { LLM } from "llama-node";
import { LLMRS } from "llama-node/dist/llm/llm-rs.js";
import path from "path";

const model = path.resolve(process.cwd(), "../ggml-alpaca-7b-q4.bin");

const llama = new LLM(LLMRS);

const template = `how are you`;

const prompt = `Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:

${template}

### Response:`;

const params: Partial<Generate> = {
    prompt,
    numPredict: 128,
    temperature: 0.2,
    topP: 1,
    topK: 40,
    repeatPenalty: 1,
    repeatLastN: 64,
    seed: 0,
    feedPrompt: true,
};

const run = async () => {
    await llama.load({ modelPath: model, modelType: ModelType.Llama });

    await llama.createCompletion(params, (response) => {
        process.stdout.write(response.token);
    });
};

run();
