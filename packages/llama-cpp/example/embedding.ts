import { LLama, LlamaContextParams, LlamaInvocation } from "../index";
import path from "path";

const llama = LLama.load(
    path.resolve(process.cwd(), "../../ggml-vicuna-7b-1.1-q4_1.bin"),
    {
        nCtx: 512,
        nParts: -1,
        seed: 0,
        f16Kv: false,
        logitsAll: false,
        vocabOnly: false,
        useMlock: false,
        embedding: true,
        useMmap: true,
    },
    false
);

const prompt = `Who is the president of the United States?`;

const params: LlamaInvocation = {
    nThreads: 4,
    nTokPredict: 2048,
    topK: 40,
    topP: 0.1,
    temp: 0.2,
    repeatPenalty: 1,
    prompt,
};

llama.getWordEmbedding(params, (data) => {
    console.log(data.data);
});
