import { LLama, Generate } from "../index";
import path from "path";

const run = async () => {
    const llama = await LLama.load(
        {
            modelPath: path.resolve(process.cwd(), "../../ggml-vic7b-q5_1.bin"),
            nCtx: 512,
            nGpuLayers: 0,
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

    const params: Generate = {
        nThreads: 4,
        nTokPredict: 2048,
        topK: 40,
        topP: 0.1,
        temp: 0.2,
        repeatPenalty: 1,
        prompt,
    };

    llama.getWordEmbedding(params).then((data) => {
        console.log(data);
    });
};

run();
