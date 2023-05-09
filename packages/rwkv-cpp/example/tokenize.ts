import { Rwkv } from "../index";
import path from "path";

const run = async () => {
    const rwkv = await Rwkv.load(
        path.resolve(
            process.cwd(),
            "../../ggml-rwkv-4_raven-7b-v9-Eng99%-20230412-ctx8192-Q4_1_0.bin"
        ),
        path.resolve(process.cwd(), "../../20B_tokenizer.json"),
        4,
        true
    );

    const template = `Who is the president of the United States?`;

    const tokens = await rwkv.tokenize(template);

    console.log(tokens);
};

run();
