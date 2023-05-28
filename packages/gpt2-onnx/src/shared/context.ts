import { loadOnnx, loadTensorFlow } from "./loader";
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
    onProgress: (data: string) => void;
}

export class IsomorphicContext {
    tokenizer?: IsomorphicTokenizer;
    session?: InferenceSession;
    tfjs?: Awaited<ReturnType<typeof loadTensorFlow>>;
    onnx?: Awaited<ReturnType<typeof loadOnnx>>;

    static async create(options: IGPT2OnnxOptions) {
        const tokenizer = options.tokenizer;
        await tokenizer.initFromUrl(options.tokenizerUrl);
        const gpt2Onnx = new IsomorphicContext();

        gpt2Onnx.onnx = await loadOnnx();
        gpt2Onnx.tfjs = await loadTensorFlow();

        gpt2Onnx.tokenizer = tokenizer;
        gpt2Onnx.session = await gpt2Onnx.onnx.InferenceSession.create(
            options.modelPath as ArrayBufferLike
        );

        return gpt2Onnx;
    }

    free() {
        this.tokenizer?.free();
    }

    getLogits(onnxTensor: TypedTensor<"float32">) {
        if (!this.tfjs) {
            throw new Error("Tensorflow not initialized");
        }
        let output = this.tfjs
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
        if (!this.tfjs) {
            throw new Error("Tensorflow not initialized");
        }
        if (!this.tokenizer) {
            throw new Error("Tokenizer not initialized");
        }

        if (!this.session) {
            throw new Error("Session not initialized");
        }

        const numPredict = inferArgs.numPredict ?? 128;
        const topK = inferArgs.topK ?? 1;

        let text = inferArgs.prompt;
        let remain = numPredict;

        while (remain > 0) {
            remain -= 1;

            // TODO: optimize to reuse input_ids and attention_mask instead of creating new ones from text
            const inputs = this.tokenizer.tokenize(text, true);
            const result = await this.session.run(inputs);

            const logits = this.getLogits(
                result["last_hidden_state"] as TypedTensor<"float32">
            );

            let probs = this.tfjs.softmax(logits, -1);

            // TODO: implement topP
            probs = probs.topk(topK, true).indices.slice(0, 1).squeeze();

            const token = probs.dataSync();

            // TODO: implement end of sentence
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

            text += tokenText;
        }
    }
}
