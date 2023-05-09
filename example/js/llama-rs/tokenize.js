import { LLM } from "llama-node";
import { LLamaRS } from "llama-node/dist/llm/llama-rs.js";
import path from "path";
const model = path.resolve(process.cwd(), "../ggml-alpaca-7b-q4.bin");
const llama = new LLM(LLamaRS);
const content = "how are you?";
const run = async () => {
    await llama.load({ path: model });
    await llama.tokenize(content).then(console.log);
};
run();
