import { LLama } from "../index.js";
import path from "path";

const model = path.resolve(process.cwd(), "./ggml-alpaca-7b-q4.bin");

const llama = LLama.create({ path: model, numCtxTokens: 4096 });

const prompt = "test";

llama.onGenerated((response) => {
    switch (response.type) {
        case "DATA": {
            process.stdout.write(response.data.token);
            if (response.data.completed) {
                llama.terminate();
            }
            break;
        }
        case "ERROR": {
            console.log(response);
            llama.terminate();
            break;
        }
    }
});

llama.inference({ prompt, numPredict: BigInt(12) });
