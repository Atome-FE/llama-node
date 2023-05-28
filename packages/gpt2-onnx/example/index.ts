import { GPT2Onnx } from "../src/node";
import path from "path";
import { AutoTokenizer } from "../src/node/tokenizer";

const modelPath = path.join(process.cwd(), "../../gpt2.onnx");

const tokenizerUrl = "https://huggingface.co/gpt2/raw/main/tokenizer.json";

const prompt = `My name is Merve and my favorite`;

const numPredict = 128;

// TODO: sample topP
const topK = 1;

const run = async () => {
    const gpt2 = await GPT2Onnx.create({
        modelPath,
        tokenizerUrl,
        tokenizer: new AutoTokenizer(),
    });

    process.stdout.write(prompt);

    await gpt2.inference({
        prompt,
        numPredict,
        topK,
        onProgress: (data) => {
            process.stdout.write(data);
        },
    });

    gpt2.free();
};

run();
