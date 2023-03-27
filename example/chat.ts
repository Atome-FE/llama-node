import { LLamaClient } from "../src";
import path from "path";

const model = path.resolve(process.cwd(), "./ggml-alpaca-7b-q4.bin");

const llama = new LLamaClient(
    {
        path: model,
        numCtxTokens: 128,
    },
    true
);

const content = "how are you?";

llama.createChatCompletion(
    {
        messages: [{ role: "user", content }],
        numPredict: BigInt(128),
        temp: 0.2,
        topP: 1,
        topK: BigInt(40),
        repeatPenalty: 1,
        repeatLastN: BigInt(64),
        seed: BigInt(0),
    },
    (response) => {
        if (!response.completed) {
            process.stdout.write(response.token);
        }
    }
);
