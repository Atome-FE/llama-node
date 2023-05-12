import {
    // EmbeddingResultType,
    InferenceResultType,
    Rwkv,
    RwkvInvocation,
} from "@llama-node/rwkv-cpp";

import { type ILLM, type LLMResult, LLMError, LLMErrorType } from "./type";

export interface LoadConfig {
    modelPath: string;
    tokenizerPath: string;
    nThreads: number;
    enableLogging: boolean;
}

export interface TokenizeArguments {
    content: string;
}

export class RwkvCpp
    implements
        ILLM<Rwkv, LoadConfig, RwkvInvocation, unknown, TokenizeArguments>
{
    instance!: Rwkv;

    async load(config: LoadConfig) {
        const { modelPath, tokenizerPath, nThreads, enableLogging } = config;
        this.instance = await Rwkv.load(
            modelPath,
            tokenizerPath,
            nThreads,
            enableLogging
        );
    }

    async createCompletion(
        params: RwkvInvocation,
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

    // embedding not implemented yet

    /* async getEmbedding(params: RwkvInvocation): Promise<number[]> {
        return new Promise<number[]>((res, rej) => {
            this.instance.getWordEmbedding(params, (response) => {
                switch (response.type) {
                    case EmbeddingResultType.Data:
                        res(response.data ?? []);
                        break;
                    case EmbeddingResultType.Error:
                        rej(new Error("Unknown Error"));
                        break;
                }
            });
        });
    }

    async getDefaultEmbedding(text: string): Promise<number[]> {
        return this.getEmbedding({
            topP: 0.1,
            temp: 0.1,
            prompt: text,
            maxPredictLength: 2048,
        });
    } */

    async tokenize(params: TokenizeArguments): Promise<number[]> {
        return await this.instance.tokenize(params.content);
    }
}
