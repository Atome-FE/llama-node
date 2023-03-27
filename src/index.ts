import {
    LLama as LLamaNode,
    LLamaConfig,
    LLamaArguments,
} from "@llama-node/core";

export interface ChatMessage {
    role: "system" | "assistant" | "user";
    content: string;
}

export type ChatParams = Omit<LLamaArguments, "prompt"> & {
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
     * not functioning actually, will change in the future
     * @param params ChatParams
     * @param callback CompletionCallback
     * @returns Promise<boolean>
     */
    createChatCompletion = (
        params: ChatParams,
        callback: CompletionCallback
    ) => {
        console.warn(
            "The create chat completion function is just a simulation of dialog, it does not provide chatting interaction"
        );
        const data = new Date().toISOString();
        const { messages, ...rest } = params;
        const prompt = `You are AI assistant, please complete a dialog, where user interacts with AI assistant. AI assistant is helpful, kind, obedient, honest, and knows its own limits. AI assistant can do programming tasks and return codes.
Knowledge cutoff: 2021-09-01
Current date: ${data}
${messages.map(({ role, content }) => `${role}: ${content}`).join("\n")}
assistant: `;
        const completionParams = Object.assign({}, rest, {
            prompt: this.createAlpacaPrompt(prompt),
        });
        return this.createTextCompletion(completionParams, callback);
    };

    /**
     * Get sentence embedding, currently end token is in rust program and set to "<end>"
     * @param params LLamaArguments
     * @returns Promise<number[]>
     */
    getEmbedding = (params: LLamaArguments) => {
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
     * Create Text Completion
     * @param params LLamaArguments
     * @param callback CompletionCallback
     * @returns Promise<boolean>
     */
    createTextCompletion = (
        params: LLamaArguments,
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
