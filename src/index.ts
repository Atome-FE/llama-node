import type { ILLM, CompletionCallback, LLMResult } from "./llm/type";
export type * from "./llm/type";

class LLM<
    Instance = any,
    LoadConfig = any,
    LLMInferenceArguments = any,
    LLMEmbeddingArguments = any,
    TokenizeArguments = any
> {
    llm: ILLM<
        Instance,
        LoadConfig,
        LLMInferenceArguments,
        LLMEmbeddingArguments,
        TokenizeArguments
    >;

    constructor(
        llm: new () => ILLM<
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
        callback: CompletionCallback,
        abortSignal?: AbortSignal
    ): Promise<LLMResult> {
        return this.llm.createCompletion(params, callback, abortSignal);
    }

    async getEmbedding(params: LLMEmbeddingArguments): Promise<number[]> {
        if (!this.llm.getEmbedding) {
            console.warn("getEmbedding not implemented for current LLM");
            return [];
        } else {
            return this.llm.getEmbedding(params);
        }
    }

    async getDefaultEmbeddings(text: string): Promise<number[]> {
        if (!this.llm.getDefaultEmbedding) {
            console.warn("getDefaultEmbedding not implemented for current LLM");
            return [];
        } else {
            return this.llm.getDefaultEmbedding(text);
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

// deprecated LLama naming in the future
export { LLM as LLama };

export { LLM };
