import { LLama } from "../index";
import path from "path";

const llama = LLama.load(
    path.resolve(process.cwd(), "../../ggml-vicuna-7b-4bit-rev1.bin"),
    null,
    false
);

const template = `Who is the president of the United States?`;

llama.tokenize(template, 2048, (data) => {
    console.log(data.data);
});
