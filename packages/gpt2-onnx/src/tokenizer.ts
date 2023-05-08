import { Tokenizer } from "@llama-node/hf-tokenizer";
import { Tensor } from "onnxruntime-node";

export class AutoTokenizer {
    tokenizer: Tokenizer;

    constructor(tokenizer: Tokenizer) {
        this.tokenizer = tokenizer;
    }

    static async fromUrl(url: string) {
        const json = await (await fetch(url)).json();

        const tokenizer = new Tokenizer(JSON.stringify(json));
        return new AutoTokenizer(tokenizer);
    }

    free() {
        this.tokenizer.free();
    }

    decode(ids: Uint32Array, skipSpecialTokens = false) {
        return this.tokenizer.decode(ids, skipSpecialTokens);
    }

    tokenize(text: string, addSpecialTokens = true) {
        const encoded = this.tokenizer.encode(text, addSpecialTokens);

        const inputIdArray = BigInt64Array.from(
            Array.from(encoded.ids).map((x) => BigInt(x))
        );

        encoded.free();

        const input_ids = new Tensor("int64", inputIdArray, [
            1,
            inputIdArray.length,
        ]);

        const attentionMaskArray = inputIdArray.map(() => BigInt(1));

        const attention_mask = new Tensor("int64", attentionMaskArray, [
            1,
            attentionMaskArray.length,
        ]);

        return { input_ids, attention_mask };
    }
}
