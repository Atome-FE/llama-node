import { LLama } from "../index";
import path from "path";
import fs from "fs";

const model = path.resolve(process.cwd(), "../../ggml-alpaca-7b-q4.bin");

LLama.enableLogger();

const llama = LLama.create({
    path: model,
    numCtxTokens: 128,
});

const prompt = `My favourite animal is the cat`;

llama.getWordEmbeddings(
    {
        prompt,
        numPredict: BigInt(128),
        temp: 0.2,
        topP: 1,
        topK: BigInt(40),
        repeatPenalty: 1,
        repeatLastN: BigInt(64),
        seed: BigInt(0),
    },
    (response) => {
        switch (response.type) {
            case "DATA": {
                // console.log(response.data);
                fs.writeFileSync(
                    path.resolve(process.cwd(), "./tmp/cat1.json"),
                    JSON.stringify(response.data)
                );
                break;
            }
            case "ERROR": {
                console.log(response);
                break;
            }
        }
    }
);
