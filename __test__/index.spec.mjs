import { LLama } from "../index.js";
import path from "path";

const obj = LLama.new();

const model = path.resolve(process.cwd(), "./ggml-alpaca-7b-q4.bin");

obj.loadModel({ path: model });

const prompt = "Hello from nodejs!";

obj.onGenerated((err, data) => {
    if (!err) {
        process.stdout.write(data.token);
        if (data.completed) {
            console.log(data);
        }
    }
});

obj.inference({ prompt });
