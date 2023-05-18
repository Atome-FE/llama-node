import { InferenceResultType, Llm, ModelType } from "../index";
import path from "path";

const model = path.resolve(process.cwd(), "../../ggml-alpaca-7b-q4.bin");
// const persistSession = path.resolve(process.cwd(), "./tmp/session.bin");

const run = async () => {
    const llm = await Llm.load(
        {
            modelType: ModelType.Llama,
            modelPath: model,
            numCtxTokens: 128,
        },
        true
    );

    const template = `how are you`;

    const prompt = `Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:

${template}

### Response:`;

    llm.inference(
        {
            prompt,
            numPredict: 128,
            temperature: 0.2,
            topP: 1,
            topK: 40,
            repeatPenalty: 1,
            repeatLastN: 64,
            seed: 0,
            feedPrompt: true,
        },
        (response) => {
            switch (response.type) {
                case InferenceResultType.Data: {
                    process.stdout.write(response.data?.token ?? "");
                    break;
                }
                case InferenceResultType.End:
                case InferenceResultType.Error: {
                    console.log(response);
                    break;
                }
            }
        }
    );
};
run();
