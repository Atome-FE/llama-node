import { LLama } from "../index";
import path from "path";

const model = path.resolve(process.cwd(), "../../ggml-alpaca-7b-q4.bin");

LLama.enableLogger();

const run = async () => {
    const llama = await LLama.create({
        path: model,
        numCtxTokens: 128,
    });

    const prompt = "My favourite animal is the cat";

    const tokens = await llama.tokenize(prompt);

    console.log(tokens);
};

run();