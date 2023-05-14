import { LLM } from "llama-node";
import { LLMRS } from "llama-node/dist/llm/llm-rs.js";
import path from "path";
const model = path.resolve(process.cwd(), "../ggml-alpaca-7b-q4.bin");
const llama = new LLM(LLMRS);
const content = "how are you?";
const run = async () => {
    await llama.load({ modelPath: model, modelType: "Llama" /* ModelType.Llama */ });
    await llama.tokenize(content).then(console.log);
};
run();
