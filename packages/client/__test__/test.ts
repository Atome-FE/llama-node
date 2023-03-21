import path from "path";
import { LLamaClient } from "../src";

const model = path.resolve(process.cwd(), "./ggml-alpaca-7b-q4.bin");

const client = new LLamaClient({ path: model, numCtxTokens: 4096 }, true);

client.createChatCompletion(
    {
        messages: [
            {
                role: "user",
                content:
                    "Give me a code example of counter program with state written in React.js, including increment button and decrement button",
            },
        ],
        numPredict: BigInt(32768),
        seed: BigInt(-1),
    },
    (data) => {
        process.stdout.write(data.token);
    }
);
