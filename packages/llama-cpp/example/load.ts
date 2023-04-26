import { LLama, LlamaInvocation } from "../index";
import path from "path";

const llama = LLama.load(
    path.resolve(process.cwd(), "../../ggml-vicuna-7b-1.1-q4_1.bin"),
    null,
    false
);

const template = `Who is the president of the United States?`;

const prompt = `A chat between a user and an assistant.
USER: ${template}
ASSISTANT:`;

const params: LlamaInvocation = {
    nThreads: 4,
    nTokPredict: 2048,
    topK: 40,
    topP: 0.1,
    temp: 0.2,
    repeatPenalty: 1,
    prompt,
};

llama.inference(params, (data) => {
    process.stdout.write(data.data?.token ?? "");
});
