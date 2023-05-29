import { Tokenizer } from "@llama-node/hf-tokenizer/nodejs/tokenizer-node";
import { Tensor } from "onnxruntime-node";
import axios from "axios";
import type { TensorConstructor } from "onnxruntime-node";

export class IsomorphicTokenizer {
    tokenizer?: Tokenizer;
    Tensor?: TensorConstructor;

    async initFromUrl(url: string) {
        const json = await axios.get(url).then((res) => res.data);

        this.Tensor = Tensor;
        this.tokenizer = new Tokenizer(JSON.stringify(json));
    }

    free() {
        this.tokenizer?.free();
    }

    decode(ids: Uint32Array, skipSpecialTokens = false) {
        return this.tokenizer?.decode(ids, skipSpecialTokens) ?? "";
    }

    toOnnx(input: bigint[]) {
        if (!this.Tensor) {
            throw new Error("Onnx is not initialized");
        }

        const inputArray = BigInt64Array.from(input);

        const input_ids = new this.Tensor("int64", inputArray, [
            1,
            inputArray.length,
        ]);

        const attentionMaskArray = inputArray.map(() => BigInt(1));

        const attention_mask = new this.Tensor("int64", attentionMaskArray, [
            1,
            attentionMaskArray.length,
        ]);

        return { input_ids, attention_mask };
    }

    tokenize(text: string, addSpecialTokens = true) {
        if (!this.Tensor) {
            throw new Error("Onnx is not initialized");
        }

        const encoded = this.tokenizer?.encode(text, addSpecialTokens);

        const inputIdArray = Array.from(encoded?.ids ?? []).map((x) =>
            BigInt(x)
        );

        encoded?.free();

        return inputIdArray;
    }
}
