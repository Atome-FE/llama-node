import path from "path";
import { LLamaClient } from "../src";

const model = path.resolve(process.cwd(), "./ggml-alpaca-7b-q4.bin");

const client = new LLamaClient({ path: model, numCtxTokens: 4096 }, true);

const prompt = `// Show an example of counter component in react.js. it has increment and decrement buttons where they change the state by 1.
export const Counter => {`;

client.createTextCompletion(
    {
        prompt,
        numPredict: BigInt(2048),
        temp: 0.2,
        topP: 0.8,
        topK: BigInt(40),
        repeatPenalty: 1,
        repeatLastN: BigInt(512),
    },
    (res) => {
        process.stdout.write(res.token);
    }
);
