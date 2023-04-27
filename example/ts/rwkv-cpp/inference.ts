import { LLM } from "llama-node";
import { RwkvCpp, LoadConfig } from "llama-node/dist/llm/rwkv-cpp.js";
import path from "path";

const modelPath = path.resolve(
    process.cwd(),
    "../ggml-rwkv-4_raven-7b-v9-Eng99%-20230412-ctx8192-Q4_1_0.bin"
);
const tokenizerPath = path.resolve(process.cwd(), "../20B_tokenizer.json");

const rwkv = new LLM(RwkvCpp);

const config: LoadConfig = {
    modelPath,
    tokenizerPath,
    nThreads: 4,
    enableLogging: true,
};

rwkv.load(config);

const template = `Who is the president of the United States?`;

const prompt = `Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction: ${template}

### Response:`;

rwkv.createCompletion(
    {
        maxPredictLength: 2048,
        topP: 0.1,
        temp: 0.1,
        prompt,
    },
    (response) => {
        process.stdout.write(response.token);
    }
);
