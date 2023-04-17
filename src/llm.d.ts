export interface CompletionCallback {
    (data: { token: string; completed: boolean }): void;
}

export interface LLM<
    Instance,
    LoadConfig,
    LLMInferenceArguments,
    LLMEmbeddingArguments,
    LLMTokenizeArguments
> {
    readonly instance: Instance;

    load(config: LoadConfig): void;

    createCompletion(
        params: LLMInferenceArguments,
        callback: CompletionCallback
    ): Promise<boolean>;

    getEmbedding?(params: LLMEmbeddingArguments): Promise<number[]>;

    getDefaultEmbedding?(text: string): Promise<number[]>;

    tokenize?(content: LLMTokenizeArguments): Promise<number[]>;
}
