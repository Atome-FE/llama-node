import init, { Tokenizer } from "@llama-node/hf-tokenizer/web";
import { Tensor } from "onnxruntime-web";
import { IsomorphicTokenizer } from "../shared/tokenizer";

export class AutoTokenizer extends IsomorphicTokenizer {
    constructor() {
        super(Tokenizer, Tensor);
    }

    async initFromUrl(url: string) {
        await init();

        await super.initFromUrl(url);
    }
}
