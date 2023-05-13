import { LLama } from "../index";
import path from "path";

const run = async () => {
    const llama = await LLama.load(
        path.resolve(process.cwd(), "../../ggml-vic7b-q5_1.bin"),
        null,
        false
    );

    const template = `Who is the president of the United States?`;

    llama.tokenize(template, 2048).then((data) => {
        console.log(data);
    });
};

run();
