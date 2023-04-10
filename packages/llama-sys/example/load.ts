import { LLama, LlamaContextParams, LlamaInvocation } from "../index";
import path from "path";

const llama = LLama.new(
    path.resolve(process.cwd(), "../../ggjt-alpaca-7b-q4.bin")
);

const template = `how are you`;

const prompt = `Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:

${template}

### Response:`;

const params: LlamaInvocation = {
    nThreads: 4,
    nTokPredict: 0,
    topK: 40,
    topP: 0.0,
    temp: 0.7,
    repeatPenalty: 1.2,
    stopSequence: "\n\n",
    prompt,
};

llama.run(params, () => void {});
