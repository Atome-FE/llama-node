import { LLama } from "../index";
import path from "path";

const llama = LLama.load(
    path.resolve(
        process.cwd(),
        "../../ggml-rwkv-4_raven-7b-v9-Eng99%-20230412-ctx8192-Q4_1_0.bin"
        // "./rwkv-sys/rwkv.cpp/tests/tiny-rwkv-660K-FP16.bin"
    ),
    path.resolve(process.cwd(), "../../20B_tokenizer.json"),
    4,
    true
);

const template = `Who is the president of the United States?`;

const prompt = `
### Human: ${template}

### Assistant:`;

const params = {
    nThreads: 4,
    nTokPredict: 2048,
    topK: 40,
    topP: 0.1,
    temp: 0.1,
    repeatPenalty: 1,
    stopSequence: "### Human",
    prompt: "hello ",
};

llama.inference(params, (data) => {
    // process.stdout.write(data.data?.token ?? "");
});
