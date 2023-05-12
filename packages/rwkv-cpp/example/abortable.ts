import { Rwkv, RwkvInvocation } from "../index";
import path from "path";

const run = async () => {
    const rwkv = await Rwkv.load(
        path.resolve(
            process.cwd(),
            "../../ggml-rwkv-4_raven-7b-v9-Eng99%-20230412-ctx8192-Q4_1_0.bin"
        ),
        path.resolve(process.cwd(), "../../20B_tokenizer.json"),
        4,
        true
    );

    const template = `Who is the president of the United States?`;

    const prompt = `Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction: ${template}

### Response:`;

    const params: RwkvInvocation = {
        maxPredictLength: 2048,
        topP: 0.1,
        temp: 0.1,
        prompt,
    };

    const abort = rwkv.inference(params, (data) => {
        process.stdout.write(data.data?.token ?? "");
    });

    setTimeout(() => {
        abort();
    }, 5000);
};

run();
