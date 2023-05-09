import { test, expect } from "vitest";
import { InferenceResultType, LLama } from "../index.js";

test(
    "inference is working",
    async () => {
        LLama.enableLogger();

        const llama = await LLama.create({
            path: process.env.model?.toString()!,
            numCtxTokens: 128,
        });

        const template = `how are you`;

        const prompt = `Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:

${template}

### Response:`;

        const promise = () =>
            new Promise<boolean>((res) => {
                llama.inference(
                    {
                        prompt,
                        numPredict: 128,
                        temp: 0.2,
                        topP: 1,
                        topK: 40,
                        repeatPenalty: 1,
                        repeatLastN: 64,
                        seed: 0,
                    },
                    (response) => {
                        switch (response.type) {
                            case InferenceResultType.Data: {
                                process.stdout.write(
                                    response.data?.token ?? ""
                                );
                                break;
                            }
                            case InferenceResultType.Error: {
                                break;
                            }
                            case InferenceResultType.End: {
                                expect(response.type).toEqual(
                                    InferenceResultType.End
                                );
                                res(response.type === InferenceResultType.End);
                                break;
                            }
                        }
                    }
                );
            });

        const r = await promise();
        expect(r).toBe(true);
    },
    { timeout: 60000 }
);
