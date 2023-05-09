import {
    EmbeddingResultType,
    InferenceResultType,
    LLama,
    LlamaContextParams,
    LlamaInvocation,
    TokenizeResultType,
} from "@llama-node/llama-cpp";

import { type ILLM, type LLMResult, LLMError } from "./type";

export interface LoadConfig extends LlamaContextParams {
    path: string;
    enableLogging: boolean;
}

export interface TokenizeArguments {
    content: string;
    nCtx: number;
}

export class LLamaCpp
    implements
        ILLM<
            LLama,
            LoadConfig,
            LlamaInvocation,
            LlamaInvocation,
            TokenizeArguments
        >
{
    instance!: LLama;

    async load(config: LoadConfig) {
        const { path, enableLogging, ...rest } = config;
        this.instance = LLama.load(path, rest, enableLogging);
    }

    async createCompletion(
        params: LlamaInvocation,
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
                            errors.push(response.message ?? "Unknown Error");
                            break;
                        }
                    }
                });
            }
        );
    }

    async getEmbedding(params: LlamaInvocation): Promise<number[]> {
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
            nThreads: 4,
            nTokPredict: 1024,
            topK: 40,
            topP: 0.1,
            temp: 0.1,
            repeatPenalty: 1,
            prompt: text,
        });
    }

    async tokenize(params: TokenizeArguments): Promise<number[]> {
        return new Promise<number[]>((res, rej) => {
            this.instance.tokenize(params.content, params.nCtx, (response) => {
                if (response.type === TokenizeResultType.Data) {
                    res(response.data);
                } else {
                    rej(new Error("Unknown Error"));
                }
            });
        });
    }
}
