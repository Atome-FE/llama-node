import { LLama } from "llama-node";
import { LLamaRS } from "llama-node/dist/llm/llama-rs.js";
import path from "path";
const model = path.resolve(process.cwd(), "../ggml-alpaca-7b-q4.bin");
const llama = new LLama(LLamaRS);
llama.load({ path: model });
const content = "how are you?";
llama.tokenize(content).then(console.log);
