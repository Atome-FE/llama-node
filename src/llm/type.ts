export interface CompletionCallback {
    (data: { token: string; completed: boolean }): void;
}

export interface ILLM<
    Instance,
    LoadConfig,
    LLMInferenceArguments,
    LLMEmbeddingArguments,
    LLMTokenizeArguments
> {
    readonly instance: Instance;

    load(config: LoadConfig): Promise<void>;

    createCompletion(
        params: LLMInferenceArguments,
        callback: CompletionCallback
    ): Promise<LLMResult>;

    getEmbedding?(params: LLMEmbeddingArguments): Promise<number[]>;

    getDefaultEmbedding?(text: string): Promise<number[]>;

    tokenize?(content: LLMTokenizeArguments): Promise<number[]>;
}

export interface LLMResult {
    tokens: string[];
    completed: boolean;
}

export class LLMError extends Error {
    public readonly tokens: string[];
    public readonly completed: boolean;

    constructor({
        message,
        tokens,
        completed,
    }: {
        message: string;
        tokens: string[];
        completed: boolean;
    }) {
        super(message);
        this.tokens = tokens;
        this.completed = completed;
    }
}
