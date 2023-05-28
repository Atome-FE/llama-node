import type { Tokenizer } from "@llama-node/hf-tokenizer";
import type { TensorConstructor } from "onnxruntime-node";
import { loadHfTokenizer, loadOnnx } from "./loader";

export class IsomorphicTokenizer {
    tokenizer?: Tokenizer;
    Tensor?: TensorConstructor;

    async initFromUrl(url: string) {
        const json = await (await fetch(url)).json();

        const { Tokenizer } = await loadHfTokenizer();
        const { Tensor } = await loadOnnx();

        this.Tensor = Tensor;
        this.tokenizer = new Tokenizer(JSON.stringify(json));
    }

    free() {
        this.tokenizer?.free();
    }

    decode(ids: Uint32Array, skipSpecialTokens = false) {
        return this.tokenizer?.decode(ids, skipSpecialTokens) ?? "";
    }

    tokenize(text: string, addSpecialTokens = true) {
        if (!this.Tensor) {
            throw new Error("Onnx is not initialized");
        }

        const encoded = this.tokenizer?.encode(text, addSpecialTokens);

        const inputIdArray = BigInt64Array.from(
            Array.from(encoded?.ids ?? []).map((x) => BigInt(x))
        );

        encoded?.free();

        const input_ids = new this.Tensor("int64", inputIdArray, [
            1,
            inputIdArray.length,
        ]);

        const attentionMaskArray = inputIdArray.map(() => BigInt(1));

        const attention_mask = new this.Tensor("int64", attentionMaskArray, [
            1,
            attentionMaskArray.length,
        ]);

        return { input_ids, attention_mask };
    }
}
