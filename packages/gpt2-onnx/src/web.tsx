import React from "react";
import { createRoot } from "react-dom/client";
import "./web/worker";
import type { ICommand } from "./web/worker";
import MyWorker from "./web/worker?worker";

const tokenizerUrl = "https://huggingface.co/gpt2/raw/main/tokenizer.json";

const worker = new MyWorker();

console.log("worker", worker);

const App = () => {
    const [inputText, setInputText] = React.useState("");
    const [inferenceText, setInferenceText] = React.useState("");

    const handleLoad = React.useCallback((model: ArrayBufferLike) => {
        const message: ICommand = {
            type: "load",
            data: {
                model,
                tokenizerUrl,
            },
        };
        worker.postMessage(message);
    }, []);

    const handleInference = React.useCallback(() => {
        const message: ICommand = {
            type: "inference",
            data: {
                prompt: inputText,
                topK: 1,
                numPredict: 128,
            },
        };
        worker.postMessage(message);
    }, [inputText]);

    const handleFileChange = React.useCallback(
        async (e: React.ChangeEvent<HTMLInputElement>) => {
            const model = await e.target.files?.[0]?.arrayBuffer();
            if (!model) {
                throw new Error("No model");
            } else {
                return handleLoad(model);
            }
        },
        []
    );

    React.useEffect(() => {
        worker.onmessage = (e) => {
            setInferenceText((text) => text + e.data);
        };
    }, []);

    return (
        <div>
            <div>
                <input onChange={handleFileChange} type="file"></input>
            </div>
            <div>
                <input
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                />
                <button onClick={handleInference}>Inference</button>
            </div>
            <div>{inferenceText}</div>
        </div>
    );
};

const root = createRoot(document.getElementById("root")!);
root.render(<App />);
