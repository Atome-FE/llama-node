import {
    InferenceResultType,
    LLama,
    ModelLoad,
    Generate,
} from "@llama-node/llama-cpp";

import { type ILLM, type LLMResult, LLMError, LLMErrorType } from "./type";

export interface LoadConfig extends ModelLoad {
    enableLogging: boolean;
}

export class LLamaCpp
    implements ILLM<LLama, LoadConfig, Generate, Generate, string>
{
    instance!: LLama;

    async load(config: LoadConfig) {
        const { enableLogging, ...rest } = config;
        this.instance = await LLama.load(rest, enableLogging);
    }

    async createCompletion(
        params: Generate,
        callback: (data: { token: string; completed: boolean }) => void,
        abortSignal?: AbortSignal
    ): Promise<LLMResult> {
        let completed = false;
        const tokens: string[] = [];
        const errors: string[] = [];
        return new Promise<LLMResult>(
            (res, rej: (reason: LLMError) => void) => {
                const abort = this.instance.inference(params, (response) => {
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
                                        type: LLMErrorType.Generic,
                                    })
                                );
                            } else {
                                res({ tokens, completed });
                            }
                            break;
                        }
                        case InferenceResultType.Error: {
                            errors.push(response.message ?? "Unknown Error");
                            break;
                        }
                    }
                });

                const abortSignalHandler = () => {
                    abort();
                    rej(
                        new LLMError({
                            message: "Aborted",
                            tokens,
                            completed,
                            type: LLMErrorType.Aborted,
                        })
                    );
                    abortSignal?.removeEventListener(
                        "abort",
                        abortSignalHandler
                    );
                };

                abortSignal?.addEventListener("abort", abortSignalHandler);
            }
        );
    }

    async getEmbedding(params: Generate): Promise<number[]> {
        return await this.instance.getWordEmbedding(params);
    }

    async getDefaultEmbedding(text: string): Promise<number[]> {
        return this.getEmbedding({
            nThreads: 4,
            nTokPredict: 1024,
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
