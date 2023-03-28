import { LLama } from "../index";
import path from "path";

const model = path.resolve(process.cwd(), "../../ggml-alpaca-7b-q4.bin");

LLama.enableLogger();

const llama = LLama.create({
    path: model,
    numCtxTokens: 128,
});

const prompt = "My favourite animal is the cat";

llama.tokenize(prompt, (response) => {
    console.log(response);
    console.log(response.data.length); // 7
});
