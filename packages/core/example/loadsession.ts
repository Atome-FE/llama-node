import { InferenceResultType, LLama } from "../index";
import path from "path";

const model = path.resolve(process.cwd(), "../../ggml-alpaca-7b-q4.bin");
const loadSession = path.resolve(process.cwd(), "./tmp/session.bin");

LLama.enableLogger();

const llama = LLama.create({
    path: model,
    numCtxTokens: 128,
});

llama.inference(
    {
        prompt: "",
        numPredict: 128,
        temp: 0.2,
        topP: 1,
        topK: 40,
        repeatPenalty: 1,
        repeatLastN: 64,
        seed: 0,
        loadSession,
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
