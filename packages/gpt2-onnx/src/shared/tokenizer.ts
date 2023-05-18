import type { Tokenizer } from "@llama-node/hf-tokenizer";
import type { TensorConstructor } from "onnxruntime-node";

export class IsomorphicTokenizer {
    tokenizer?: Tokenizer;

    constructor(
        private tokenizerConstructor: new (json: string) => Tokenizer,
        private tensorConstructor: TensorConstructor
    ) {}

    async initFromUrl(url: string) {
        const json = await (await fetch(url)).json();

        this.tokenizer = new this.tokenizerConstructor(JSON.stringify(json));
    }

    free() {
        this.tokenizer?.free();
    }

    decode(ids: Uint32Array, skipSpecialTokens = false) {
        return this.tokenizer?.decode(ids, skipSpecialTokens) ?? "";
    }

    tokenize(text: string, addSpecialTokens = true) {
        const encoded = this.tokenizer?.encode(text, addSpecialTokens);

        const inputIdArray = BigInt64Array.from(
            Array.from(encoded?.ids ?? []).map((x) => BigInt(x))
        );

        encoded?.free();

        const input_ids = new this.tensorConstructor("int64", inputIdArray, [
            1,
            inputIdArray.length,
        ]);

        const attentionMaskArray = inputIdArray.map(() => BigInt(1));

        const attention_mask = new this.tensorConstructor(
            "int64",
            attentionMaskArray,
            [1, attentionMaskArray.length]
        );

        return { input_ids, attention_mask };
    }
}
