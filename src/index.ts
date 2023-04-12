import { CompletionCallback } from "./llm";
import { LLM } from "./llm";

export class LLama<
    Instance,
    LoadConfig,
    LLMInferenceArguments,
    LLMEmbeddingArguments,
    TokenizeArguments
> {
    llm: LLM<
        Instance,
        LoadConfig,
        LLMInferenceArguments,
        LLMEmbeddingArguments,
        TokenizeArguments
    >;

    constructor(
        llm: new () => LLM<
            Instance,
            LoadConfig,
            LLMInferenceArguments,
            LLMEmbeddingArguments,
            TokenizeArguments
        >
    ) {
        this.llm = new llm();
    }

    load(config: LoadConfig) {
        return this.llm.load(config);
    }

    async createCompletion(
        params: LLMInferenceArguments,
        callback: CompletionCallback
    ): Promise<boolean> {
        return this.llm.createCompletion(params, callback);
    }

    async getEmbedding(params: LLMEmbeddingArguments): Promise<number[]> {
        if (!this.llm.getEmbedding) {
            console.warn("getEmbedding not implemented for current LLM");
            return [];
        } else {
            return this.llm.getEmbedding(params);
        }
    }

    async tokenize(content: TokenizeArguments): Promise<number[]> {
        if (!this.llm.tokenize) {
            console.warn("tokenize not implemented for current LLM");
            return [];
        } else {
            return this.llm.tokenize(content);
        }
    }
}
