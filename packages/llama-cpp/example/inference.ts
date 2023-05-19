import { InferenceResultType } from "../index";
import { LLama, Generate } from "../index";
import path from "path";

const run = async () => {
    const llama = await LLama.load(
        {
            modelPath: path.resolve(process.cwd(), "../../ggml-vic7b-q5_1.bin"),
            nGpuLayers: 32,
            nCtx: 1024,
            seed: 0,
            f16Kv: false,
            logitsAll: false,
            vocabOnly: false,
            useMlock: false,
            embedding: false,
            useMmap: true,
        },
        true
    );

    const template = `Who is the president of the United States?`;

    const prompt = `A chat between a user and an assistant.
USER: ${template}
ASSISTANT:`;

    const params: Generate = {
        nThreads: 4,
        nTokPredict: 2048,
        topK: 40,
        topP: 0.1,
        temp: 0.2,
        repeatPenalty: 1,
        prompt,
    };

    const start = Date.now();

    let count = 0;
    llama.inference(params, (data) => {
        count += 1;
        process.stdout.write(data.data?.token ?? "");
        if (data.type === InferenceResultType.End) {
            const end = Date.now();
            console.log(`\n\nToken Count: ${count}`);
            console.log(`\n\nTime: ${end - start}ms`);
        }
    });
};

run();
