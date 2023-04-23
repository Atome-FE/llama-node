import { LLama } from "llama-node";
import { LLamaRS } from "llama-node/dist/llm/llama-rs.js";
import path from "path";
import fs from "fs";
const model = path.resolve(process.cwd(), "../ggml-alpaca-7b-q4.bin");
const llama = new LLama(LLamaRS);
llama.load({ path: model });
const getWordEmbeddings = async (prompt, file) => {
    const data = await llama.getEmbedding({
        prompt,
        numPredict: 128,
        temp: 0.2,
        topP: 1,
        topK: 40,
        repeatPenalty: 1,
        repeatLastN: 64,
        seed: 0,
    });
    console.log(prompt, data);
    await fs.promises.writeFile(path.resolve(process.cwd(), file), JSON.stringify(data));
};
const run = async () => {
    const dog1 = `My favourite animal is the dog`;
    await getWordEmbeddings(dog1, "./example/semantic-compare/dog1.json");
    const dog2 = `I have just adopted a cute dog`;
    await getWordEmbeddings(dog2, "./example/semantic-compare/dog2.json");
    const cat1 = `My favourite animal is the cat`;
    await getWordEmbeddings(cat1, "./example/semantic-compare/cat1.json");
};
run();
