import { test, expect } from "vitest";
import { LLama } from "../index.js";

test(
    "inference is working",
    async () => {
        LLama.enableLogger();

        const llama = LLama.create({
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
                        numPredict: BigInt(128),
                        temp: 0.2,
                        topP: 1,
                        topK: BigInt(40),
                        repeatPenalty: 1,
                        repeatLastN: BigInt(64),
                        seed: BigInt(0),
                    },
                    (response) => {
                        switch (response.type) {
                            case "DATA": {
                                process.stdout.write(response.data.token);
                                break;
                            }
                            case "END":
                            case "ERROR": {
                                expect(response.type).toEqual("END");
                                res(response.type === "END");
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
