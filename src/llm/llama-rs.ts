import { LLama, LLamaConfig, LLamaInferenceArguments } from "@llama-node/core";
import type { LLM } from "../llm";

export class LLamaRS
    implements
        LLM<
            LLama,
            LLamaConfig,
            LLamaInferenceArguments,
            LLamaInferenceArguments,
            string
        >
{
    instance!: LLama;

    load(config: LLamaConfig) {
        this.instance = LLama.create(config);
    }

    async createCompletion(
        params: LLamaInferenceArguments,
        callback: (data: { token: string; completed: boolean }) => void
    ): Promise<boolean> {
        let completed = false;
        const errors: string[] = [];
        return new Promise<boolean>((res, rej) => {
            this.instance.inference(params, (response) => {
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
    }

    async getEmbedding(params: LLamaInferenceArguments): Promise<number[]> {
        return new Promise<number[]>((res, rej) => {
            this.instance.getWordEmbeddings(params, (response) => {
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
    }

    async tokenize(params: string): Promise<number[]> {
        return new Promise<number[]>((res) => {
            this.instance.tokenize(params, (response) => {
                res(response.data);
            });
        });
    }
}
