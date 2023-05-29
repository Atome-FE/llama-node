import { GPT2Onnx } from "./index";
import { AutoTokenizer } from "./tokenizer";

export interface IInferenceCommand {
    type: "inference";
    data: {
        prompt: string;
        numPredict: number;
        topK: number;
    };
}

export interface ILoadCommand {
    type: "load";
    data: {
        model: ArrayBufferLike;
        tokenizerUrl: string;
    };
}

export type ICommand = IInferenceCommand | ILoadCommand;

class Context {
    gpt2?: GPT2Onnx;

    load = async (data: ILoadCommand["data"]) => {
        this.gpt2 = await GPT2Onnx.create({
            modelPath: data.model,
            tokenizerUrl: data.tokenizerUrl,
            tokenizer: new AutoTokenizer(),
        });
    };

    inference = async (data: IInferenceCommand["data"]) => {
        const { prompt, numPredict, topK } = data;
        if (this.gpt2) {
            await this.gpt2.inference({
                prompt,
                numPredict,
                topK,
                onProgress: (data) => {
                    postMessage(data);
                },
            });
        }
    };
}

const context = new Context();

onmessage = (event: MessageEvent<ICommand>) => {
    if (event.data.type === "load") {
        context.load(event.data.data);
    }
    if (event.data.type === "inference") {
        context.inference(event.data.data);
    }
};
