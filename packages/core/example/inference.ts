import { LLama } from "../index";
import path from "path";

const model = path.resolve(process.cwd(), "../../ggml-alpaca-7b-q4.bin");

LLama.enableLogger();

const llama = LLama.create({
    path: model,
    numCtxTokens: 128,
});

const template = `how are you`;

const prompt = `Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:

${template}

### Response:`

llama.inference(
    {
        prompt,
        numPredict: BigInt(128),
        temp: 0.2,
        topP: 1,
        topK: BigInt(40),
        repeatPenalty: 1,
        repeatLastN: BigInt(64),
        seed: BigInt(0),
    },
    (response) => {
        switch (response.type) {
            case "DATA": {
                process.stdout.write(response.data.token);
                break;
            }
            case "END":
            case "ERROR": {
                console.log(response);
                break;
            }
        }
    }
);
