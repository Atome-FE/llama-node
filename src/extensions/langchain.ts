import { AsyncCaller } from "langchain/dist/util/async_caller";
import { Embeddings, type EmbeddingsParams } from "langchain/embeddings/base";
import type { LLama } from "..";

export class LLamaEmbeddings implements Embeddings {
    caller: AsyncCaller;
    llm: LLama;

    constructor(params: EmbeddingsParams, llm: LLama) {
        if ((params.maxConcurrency ?? 1) > 1) {
            console.warn(
                "maxConcurrency > 1 not officially supported for llama-node, use at your own risk"
            );
        }
        this.caller = new AsyncCaller(params);
        this.llm = llm;
    }

    embedDocuments(documents: string[]): Promise<number[][]> {
        const promises = documents.map((doc) =>
            this.llm.getDefaultEmbeddings(doc)
        );
        return Promise.all(promises);
    }

    embedQuery(document: string): Promise<number[]> {
        return this.llm.getDefaultEmbeddings(document);
    }
}
