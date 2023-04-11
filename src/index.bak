import {
    LLama as LLamaNode,
    LLamaConfig,
    LLamaInferenceArguments,
} from "@llama-node/core";

export interface ChatMessage {
    role: "assistant" | "user";
    content: string;
}

export type ChatParams = Omit<LLamaInferenceArguments, "prompt"> & {
    messages: ChatMessage[];
};

export interface CompletionCallback {
    (data: { token: string; completed: boolean }): void;
}

export class LLamaClient {
    private llamaNode: LLamaNode;

    constructor(config: LLamaConfig, enableLogger?: boolean) {
        if (enableLogger) {
            LLamaNode.enableLogger();
        }
        this.llamaNode = LLamaNode.create(config);
    }

    /**
     * Create a prompt from alpaca template, can do instruct on alpaca model
     * @param prompt string
     * @returns string
     */
    createAlpacaPrompt = (
        prompt: string
    ) => `Below is an instruction that describes a task. Write a response that appropriately completes the request.
    
    ### Instruction:
    
    ${prompt}
    
    ### Response:`;

    /**
     * wanting to create chat completion similar to openai's chatgpt
     * not sure it is functioning, may change in the future
     * @param params ChatParams
     * @param callback CompletionCallback
     * @returns Promise<boolean>
     */
    createChatCompletion = (
        params: ChatParams,
        callback: CompletionCallback
    ) => {
        const { messages, feedPrompt = true, ...rest } = params;
        if (messages[messages.length - 1]?.role === "assistant") {
            console.warn("ChatMessage must end with user instruction");
        }
        const prompt = `Below is an instruction that describes a task. Write a response that appropriately completes the request.

${messages
    .map(
        ({ role, content }) =>
            `${
                role === "user" ? "### Instruction:\n" : "### Response:\n"
            }${content}\n`
    )
    .join("\n")}
### Response:\n`;
        const completionParams = Object.assign({}, rest, {
            prompt,
            feedPrompt,
        });
        return this.createTextCompletion(completionParams, callback);
    };

    /**
     * Get sentence embedding, currently end token is in rust program and set to "<end>"
     * @param params LLamaInferenceArguments
     * @returns Promise<number[]>
     */
    getEmbedding = (params: LLamaInferenceArguments) => {
        return new Promise<number[]>((res, rej) => {
            this.llamaNode.getWordEmbeddings(params, (response) => {
                switch (response.type) {
                    case "DATA":
                        res(response.data ?? []);
                        break;
                    case "ERROR":
                        rej(response.message);
                        break;
                }
            });
        });
    };

    /**
     * Get tokenization from string
     * @param params string
     * @returns Promise<number[]>
     */
    tokenize = (params: string) => {
        return new Promise<number[]>((res) => {
            this.llamaNode.tokenize(params, (response) => {
                res(response.data);
            });
        });
    };

    /**
     * Create Text Completion
     * @param params LLamaInferenceArguments
     * @param callback CompletionCallback
     * @returns Promise<boolean>
     */
    createTextCompletion = (
        params: LLamaInferenceArguments,
        callback: CompletionCallback
    ) => {
        let completed = false;
        const errors: string[] = [];
        return new Promise<boolean>((res, rej) => {
            this.llamaNode.inference(params, (response) => {
                switch (response.type) {
                    case "DATA": {
                        const data = {
                            token: response.data.token,
                            completed: !!response.data.completed,
                        };
                        if (data.completed) {
                            completed = true;
                        }
                        callback(data);
                        break;
                    }
                    case "END": {
                        if (errors.length) {
                            rej(new Error(errors.join("\n")));
                        } else {
                            res(completed);
                        }
                        break;
                    }
                    case "ERROR": {
                        errors.push(response.message);
                        break;
                    }
                }
            });
        });
    };
}
