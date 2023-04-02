import { LLamaClient } from "../src";
import path from "path";

const model = path.resolve(process.cwd(), "./ggml-alpaca-7b-q4.bin");

const llama = new LLamaClient(
    {
        path: model,
        numCtxTokens: 1024,
    },
    true
);

const content = `interface IPerson {
    name: string;
    email: string;
    gender: 'M' | 'F'
}
  
generate a react form that create a person in the type of the above IPerson, use Ant Design and Typescript`;

llama.createChatCompletion(
    {
        messages: [{ role: "user", content }],
        numPredict: 1024,
        temp: 0.2,
        topP: 0.4,
        topK: 40,
        repeatPenalty: 1,
        repeatLastN: 64,
        seed: 0,
    },
    (response) => {
        // if (!response.completed) {
            process.stdout.write(response.token);
        // }
    }
);
