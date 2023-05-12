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
        callback: CompletionCallback,
        abortSignal?: AbortSignal
    ): Promise<LLMResult>;

    getEmbedding?(params: LLMEmbeddingArguments): Promise<number[]>;

    getDefaultEmbedding?(text: string): Promise<number[]>;

    tokenize?(content: LLMTokenizeArguments): Promise<number[]>;
}

export interface LLMResult {
    tokens: string[];
    completed: boolean;
}

export enum LLMErrorType {
    Aborted = "Aborted",
    Generic = "Generic",
}

export class LLMError extends Error {
    public readonly tokens: string[];
    public readonly completed: boolean;
    public readonly type: LLMErrorType;

    constructor({
        message,
        tokens,
        completed,
        type,
    }: {
        message: string;
        tokens: string[];
        completed: boolean;
        type: LLMErrorType;
    }) {
        super(message);
        this.tokens = tokens;
        this.completed = completed;
        this.type = type;
    }
}
