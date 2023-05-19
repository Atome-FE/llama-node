import { LLama, Generate } from "../index";
import path from "path";

const run = async () => {
    const llama = await LLama.load(
        {
            modelPath: path.resolve(process.cwd(), "../../ggml-vic7b-q5_1.bin"),
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

    const abort = llama.inference(params, (data) => {
        process.stdout.write(data.data?.token ?? "");
    });

    setTimeout(() => {
        abort();
    }, 3000);
};

run();
