import { LLama } from "../../src";
import { LLamaRS } from "../../src/llm/llama-rs";
import path from "path";

const model = path.resolve(process.cwd(), "./ggml-alpaca-7b-q4.bin");

const llama = new LLama(LLamaRS);

llama.load({ path: model });

const content = "how are you?";

llama.tokenize(content).then(console.log);
