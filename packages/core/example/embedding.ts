import { LLM, Llm, ModelType } from "../index";
import path from "path";
import fs from "fs";

const model = path.resolve(process.cwd(), "../../ggml-alpaca-7b-q4.bin");

const getWordEmbeddings = async (llm: LLM, prompt: string, file: string) => {
    const response = await llm.getWordEmbeddings({
        prompt,
        numPredict: 128,
        temperature: 0.2,
        topP: 1,
        topK: 40,
        repeatPenalty: 1,
        repeatLastN: 64,
        seed: 0,
    });

    fs.writeFileSync(
        path.resolve(process.cwd(), file),
        JSON.stringify(response)
    );
};

const run = async () => {
    const llm = await Llm.load(
        {
            modelType: ModelType.Llama,
            modelPath: model,
            numCtxTokens: 128,
        },
        true
    );

    const dog1 = `My favourite animal is the dog`;
    getWordEmbeddings(llm, dog1, "./example/semantic-compare/dog1.json");

    const dog2 = `I have just adopted a cute dog`;
    getWordEmbeddings(llm, dog2, "./example/semantic-compare/dog2.json");

    const cat1 = `My favourite animal is the cat`;
    getWordEmbeddings(llm, cat1, "./example/semantic-compare/cat1.json");
};

run();
