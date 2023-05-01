import { InferenceResultType, LLama } from "../index";
import path from "path";

const model = path.resolve(process.cwd(), "../../ggml-alpaca-7b-q4.bin");
const saveSession = path.resolve(process.cwd(), "./tmp/session.bin");

LLama.enableLogger();

const llama = LLama.create({
    path: model,
    numCtxTokens: 128,
});

const template = `how are you`;

const prompt = `Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:

${template}

### Response:`;

llama.inference(
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
        feedPromptOnly: true,
        saveSession,
    },
    (response) => {
        switch (response.type) {
            case InferenceResultType.Data: {
                process.stdout.write(response.data?.token ?? "");
                break;
            }
            case InferenceResultType.End:
            case InferenceResultType.Error: {
                console.log(response);
                break;
            }
        }
    }
);
