import { LLama } from "../../src";
import { LLamaRS } from "../../src/llm/llama-rs";
import path from "path";

const model = path.resolve(process.cwd(), "./ggml-alpaca-7b-q4.bin");

const llama = new LLama(LLamaRS);

llama.load({ path: model });

const template = `how are you`;

const prompt = `Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:

${template}

### Response:`;

llama.createCompletion(
    {
        prompt,
        numPredict: 128,
        temp: 0.2,
        topP: 1,
        topK: 40,
        repeatPenalty: 1,
        repeatLastN: 64,
        seed: 0,
        feedPrompt: true,
    },
    (response) => {
        process.stdout.write(response.token);
    }
);
