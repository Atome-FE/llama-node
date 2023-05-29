import init from "@llama-node/hf-tokenizer/web";
import { IsomorphicTokenizer } from "../shared/tokenizer";

export class AutoTokenizer extends IsomorphicTokenizer {
    async initFromUrl(url: string) {
        await init();
        await super.initFromUrl(url);
    }
}
