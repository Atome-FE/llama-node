import { LLama } from "../index";
import path from "path";
import fs from "fs";

const model = path.resolve(process.cwd(), "../../ggml-alpaca-7b-q4.bin");

LLama.enableLogger();

const llama = LLama.create({
    path: model,
    numCtxTokens: 128,
});

const getWordEmbeddings = (prompt: string, file: string) => {
    llama.getWordEmbeddings(
        {
            prompt,
            numPredict: 128,
            temp: 0.2,
            topP: 1,
            topK: 40,
            repeatPenalty: 1,
            repeatLastN: 64,
            seed: 0,
        },
        (response) => {
            switch (response.type) {
                case "DATA": {
                    fs.writeFileSync(
                        path.resolve(process.cwd(), file),
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
};

const dog1 = `My favourite animal is the dog`;
getWordEmbeddings(dog1, "./example/semantic-compare/dog1.json");

const dog2 = `I have just adopted a cute dog`;
getWordEmbeddings(dog2, "./example/semantic-compare/dog2.json");

const cat1 = `My favourite animal is the cat`;
getWordEmbeddings(cat1, "./example/semantic-compare/cat1.json");
