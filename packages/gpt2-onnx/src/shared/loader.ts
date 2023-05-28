export const isServer = typeof window === "undefined";

export const loadTensorFlow = async () => {
    if (isServer) {
        return await import("@tensorflow/tfjs-node");
    } else {
        return await import("@tensorflow/tfjs");
    }
};

export const loadOnnx = async () => {
    if (isServer) {
        return (await import("onnxruntime-node")).default;
    } else {
        return (await import("onnxruntime-web")).default;
    }
};

export const loadHfTokenizer = async () => {
    if (isServer) {
        return await import("@llama-node/hf-tokenizer/nodejs/tokenizer-node");
    } else {
        return await import("@llama-node/hf-tokenizer/web/tokenizer-web");
    }
};
