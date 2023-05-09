import {
    InferenceResultType,
    LLama,
    LLamaConfig,
    LLamaInferenceArguments,
} from "@llama-node/core";

import { type ILLM, type LLMResult, LLMError } from "./type";

export class LLamaRS
    implements
        ILLM<
            LLama,
            LLamaConfig,
            LLamaInferenceArguments,
            LLamaInferenceArguments,
            string
        >
{
    instance!: LLama;

    async load(config: LLamaConfig) {
        this.instance = await LLama.create(config);
    }

    async createCompletion(
        params: LLamaInferenceArguments,
        callback: (data: { token: string; completed: boolean }) => void
    ): Promise<LLMResult> {
        let completed = false;
        const tokens: string[] = [];
        const errors: string[] = [];
        return new Promise<LLMResult>(
            (res, rej: (reason: LLMError) => void) => {
                this.instance.inference(params, (response) => {
                    switch (response.type) {
                        case InferenceResultType.Data: {
                            const data = {
                                token: response.data!.token,
                                completed: !!response.data!.completed,
                            };
                            tokens.push(data.token);
                            if (data.completed) {
                                completed = true;
                            }
                            callback(data);
                            break;
                        }
                        case InferenceResultType.End: {
                            if (errors.length) {
                                rej(
                                    new LLMError({
                                        message: errors.join("\n"),
                                        tokens,
                                        completed,
                                    })
                                );
                            } else {
                                res({ tokens, completed });
                            }
                            break;
                        }
                        case InferenceResultType.Error: {
                            errors.push(response.message ?? "Unknown error");
                            break;
                        }
                    }
                });
            }
        );
    }

    async getEmbedding(params: LLamaInferenceArguments): Promise<number[]> {
        return await this.instance.getWordEmbeddings(params);
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
        return await this.instance.tokenize(params);
    }
}
