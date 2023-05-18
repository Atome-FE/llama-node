import { Llm, ModelType } from "../index";
import path from "path";

const model = path.resolve(process.cwd(), "../../ggml-alpaca-7b-q4.bin");

const run = async () => {
    const llm = await Llm.load(
        {
            modelType: ModelType.Llama,
            modelPath: model,
            numCtxTokens: 128,
        },
        true
    );

    const prompt = "My favourite animal is the cat";

    const tokens = await llm.tokenize(prompt);

    console.log(tokens);
};

run();
