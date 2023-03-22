/// <reference types="vitest" />
import { defineConfig } from "vite";
import path from "path";

const dirname = path.resolve();
export default defineConfig({
    test: {
        env: {
            model: path.resolve(dirname, "../../ggml-alpaca-7b-q4.bin"),
        },
        environment: "node",
        reporters: "verbose",
        watch: false,
    },
});
