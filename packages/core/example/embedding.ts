import { EmbeddingResultType, LLama } from "../index";
import path from "path";
import fs from "fs";

const model = path.resolve(process.cwd(), "../../ggml-alpaca-7b-q4.bin");

LLama.enableLogger();

const getWordEmbeddings = async (
    llama: LLama,
    prompt: string,
    file: string
) => {
    const response = await llama.getWordEmbeddings({
        prompt,
        numPredict: 128,
        temp: 0.2,
        topP: 1,
        topK: 40,
        repeatPenalty: 1,
        repeatLastN: 64,
        seed: 0,
    });

    switch (response.type) {
        case EmbeddingResultType.Data: {
            fs.writeFileSync(
                path.resolve(process.cwd(), file),
                JSON.stringify(response.data)
            );
            break;
        }
        case EmbeddingResultType.Error: {
            console.log(response);
            break;
        }
    }
};

const run = async () => {
    const llama = await LLama.create({
        path: model,
        numCtxTokens: 128,
    });

    const dog1 = `My favourite animal is the dog`;
    getWordEmbeddings(llama, dog1, "./example/semantic-compare/dog1.json");

    const dog2 = `I have just adopted a cute dog`;
    getWordEmbeddings(llama, dog2, "./example/semantic-compare/dog2.json");

    const cat1 = `My favourite animal is the cat`;
    getWordEmbeddings(llama, cat1, "./example/semantic-compare/cat1.json");
};

run();
