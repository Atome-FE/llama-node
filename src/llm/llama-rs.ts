import {
    EmbeddingResultType,
    InferenceResultType,
    LLama,
    LLamaConfig,
    LLamaInferenceArguments,
} from "@llama-node/core";
import type { LLM } from "../llm";

export class LLamaRS
    implements
        LLM<
            LLama,
            LLamaConfig,
            LLamaInferenceArguments,
            LLamaInferenceArguments,
            string
        >
{
    instance!: LLama;

    load(config: LLamaConfig) {
        this.instance = LLama.create(config);
    }

    async createCompletion(
        params: LLamaInferenceArguments,
        callback: (data: { token: string; completed: boolean }) => void
    ): Promise<boolean> {
        let completed = false;
        const errors: string[] = [];
        return new Promise<boolean>((res, rej) => {
            this.instance.inference(params, (response) => {
                switch (response.type) {
                    case InferenceResultType.Data: {
                        const data = {
                            token: response.data!.token,
                            completed: !!response.data!.completed,
                        };
                        if (data.completed) {
                            completed = true;
                        }
                        callback(data);
                        break;
                    }
                    case InferenceResultType.Error: {
                        if (errors.length) {
                            rej(new Error(errors.join("\n")));
                        } else {
                            res(completed);
                        }
                        break;
                    }
                    case InferenceResultType.End: {
                        errors.push(response.message ?? "Unknown error");
                        break;
                    }
                }
            });
        });
    }

    async getEmbedding(params: LLamaInferenceArguments): Promise<number[]> {
        return new Promise<number[]>((res, rej) => {
            this.instance.getWordEmbeddings(params, (response) => {
                switch (response.type) {
                    case EmbeddingResultType.Data:
                        res(response.data ?? []);
                        break;
                    case EmbeddingResultType.Error:
                        rej(response.message);
                        break;
                }
            });
        });
    }

    async getDefaultEmbedding(text: string): Promise<number[]> {
        return this.getEmbedding({
            nThreads: 4,
            numPredict: 1024,
            topK: 40,
            topP: 0.1,
            temp: 0.1,
            repeatPenalty: 1,
            prompt: text,
        });
    }

    async tokenize(params: string): Promise<number[]> {
        return new Promise<number[]>((res) => {
            this.instance.tokenize(params, (response) => {
                res(response.data);
            });
        });
    }
}
