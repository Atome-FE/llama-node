import { LLama } from "../index";
import path from "path";

const run = async () => {
    const llama = await LLama.load(
        {
            modelPath: path.resolve(process.cwd(), "../../ggml-vic7b-q5_1.bin"),
        },
        false
    );

    const template = `Who is the president of the United States?`;

    llama.tokenize(template).then((data) => {
        console.log(data);
    });
};

run();
