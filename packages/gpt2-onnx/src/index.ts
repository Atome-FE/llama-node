import * as onnx from "onnxruntime-node";
import * as tfjs from "@tensorflow/tfjs-node";
import { AutoTokenizer } from "./tokenizer";

interface IGPT2OnnxOptions {
    modelPath: string;
    tokenizerUrl: string;
}

interface IGPT2OnnxInferenceOptions {
    numPredict?: number;
    prompt: string;
    topK?: number;
    endToken?: number;
    onProgress: (data: string) => void;
}

export class GPT2Onnx {
    tokenizer: AutoTokenizer | undefined;
    session: onnx.InferenceSession | undefined;

    static async create(options: IGPT2OnnxOptions) {
        const gpt2Onnx = new GPT2Onnx();
        gpt2Onnx.tokenizer = await AutoTokenizer.fromUrl(options.tokenizerUrl);
        gpt2Onnx.session = await onnx.InferenceSession.create(
            options.modelPath
        );

        return gpt2Onnx;
    }

    free() {
        this.tokenizer?.free();
    }

    getLogits(onnxTensor: onnx.TypedTensor<"float32">) {
        let output = tfjs
            .tensor<tfjs.Rank.R3>(onnxTensor.data, onnxTensor.dims as any)
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
        const topK = inferArgs.topK ?? 1;

        let text = inferArgs.prompt;
        let remain = numPredict;

        while (remain > 0) {
            remain -= 1;

            const inputs = this.tokenizer.tokenize(text, true);

            const result = await this.session.run(inputs);

            const logits = this.getLogits(
                result["last_hidden_state"] as onnx.TypedTensor<"float32">
            );

            let probs = tfjs.softmax(logits, -1);

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
