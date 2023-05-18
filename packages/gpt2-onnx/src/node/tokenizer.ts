import { Tokenizer } from "@llama-node/hf-tokenizer";
import { Tensor } from "onnxruntime-node";
import { IsomorphicTokenizer } from "../shared/tokenizer";

export class AutoTokenizer extends IsomorphicTokenizer {
    constructor() {
        super(Tokenizer, Tensor);
    }
}
