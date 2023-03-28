import { LLamaClient } from "../src";
import path from "path";

const model = path.resolve(process.cwd(), "./ggml-alpaca-7b-q4.bin");

const llama = new LLamaClient(
    {
        path: model,
        numCtxTokens: 128,
    },
    true
);

const content = "how are you?";

llama.tokenize(content).then(console.log);
