import { LLM } from "llama-node";
import { RwkvCpp } from "llama-node/dist/llm/rwkv-cpp.js";
import path from "path";
const modelPath = path.resolve(process.cwd(), "../ggml-rwkv-4_raven-7b-v9-Eng99%-20230412-ctx8192-Q4_1_0.bin");
const tokenizerPath = path.resolve(process.cwd(), "../20B_tokenizer.json");
const rwkv = new LLM(RwkvCpp);
const config = {
    modelPath,
    tokenizerPath,
    nThreads: 4,
    enableLogging: true,
};
rwkv.load(config);
rwkv.tokenize({ content: "hello world" }).then(console.log);
