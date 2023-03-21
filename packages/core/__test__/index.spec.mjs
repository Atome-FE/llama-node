import { LLama } from "../index.js";
import path from "path";

const model = path.resolve(process.cwd(), "./ggml-alpaca-7b-q4.bin");

const run = () => {
    LLama.enableLogger();

    const llama = LLama.create({ path: model });

    let prompt = "test";

    llama.inference({ prompt }, (response) => {
        switch (response.type) {
            case "DATA": {
                process.stdout.write(response.data.token);
                break;
            }
            case "END":
            case "ERROR": {
                console.log(response);
                break;
            }
        }
    });
};

run();
