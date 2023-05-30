import * as tfjs from "@tensorflow/tfjs-node";
import * as onnx from "onnxruntime-node";
import type { Rank } from "@tensorflow/tfjs";
import type { IsomorphicTokenizer } from "./tokenizer";
import type { InferenceSession, TypedTensor } from "onnxruntime-node";

interface IGPT2OnnxOptions {
    tokenizer: IsomorphicTokenizer;
    modelPath: string | ArrayBufferLike;
    tokenizerUrl: string;
}

interface IGPT2OnnxInferenceOptions {
    numPredict?: number;
    prompt: string;
    topK?: number;
    endToken?: number;
    seed?: number;
    onProgress: (data: string) => void;
}

export class IsomorphicContext {
    tokenizer?: IsomorphicTokenizer;
    session?: InferenceSession;

    static async create(options: IGPT2OnnxOptions) {
        const tokenizer = options.tokenizer;
        await tokenizer.initFromUrl(options.tokenizerUrl);
        const gpt2Onnx = new IsomorphicContext();

        gpt2Onnx.tokenizer = tokenizer;
        gpt2Onnx.session = await onnx.InferenceSession.create(
            options.modelPath as ArrayBufferLike
        );

        return gpt2Onnx;
    }

    free() {
        this.tokenizer?.free();
    }

    getLogits(onnxTensor: TypedTensor<"float32">) {
        let output = tfjs
            .tensor<Rank.R3>(onnxTensor.data, onnxTensor.dims as any)
            .slice(0, 1);

        return output
            .slice(
                [0, output.shape[1] - 1, 0],
                [output.shape[0], 1, output.shape[2]]
            )
            .squeeze();
    }

    async inference(inferArgs: IGPT2OnnxInferenceOptions) {
        if (!this.tokenizer) {
            throw new Error("Tokenizer not initialized");
        }

        if (!this.session) {
            throw new Error("Session not initialized");
        }

        const numPredict = inferArgs.numPredict ?? 128;
        const topK = inferArgs.topK ?? 40;

        let remain = numPredict;
        const tokens = this.tokenizer.tokenize(inferArgs.prompt, true);

        while (remain > 0) {
            remain -= 1;

            const inputs = this.tokenizer.toOnnx(tokens);
            const result = await this.session.run(inputs);

            let logits = this.getLogits(
                result["last_hidden_state"] as TypedTensor<"float32">
            );

            logits = tfjs.softmax(logits, -1);

            // get topK probs
            const probs = logits.topk(topK, true);

            // use seed to get uniform random number
            const randomToken = tfjs
                .randomUniformInt([1], 0, probs.values.size - 1, inferArgs.seed)
                .dataSync()[0];

            // get token from probs
            const token = probs.indices
                .slice([randomToken], [1])
                .squeeze()
                .dataSync();

            // TODO: implement end of sentence, not sure gpt2 has this
            if (
                token[0] >= 50256 ||
                token[0] === 0 ||
                token[0] === 1 ||
                (inferArgs.endToken && token[0] === inferArgs.endToken)
            ) {
                break;
            }

            const tokenText = this.tokenizer.decode(
                Uint32Array.from(token),
                true
            );

            inferArgs.onProgress(tokenText);
            tokens.push(BigInt(token[0]));
        }
    }
}
