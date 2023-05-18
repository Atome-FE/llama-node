import { defineConfig } from "vite";
import wasm from "vite-plugin-wasm";
import topLevelAwait from "vite-plugin-top-level-await";
import react from '@vitejs/plugin-react'
import copy from 'rollup-plugin-copy'

export default defineConfig({
    plugins: [
        react(),
        wasm(),
        topLevelAwait(),
        copy({
            verbose: true,
            hook: "buildStart",
            targets: [
                {
                    src: "node_modules/onnxruntime-web/dist/*.wasm",
                    dest: "public",
                },
            ],
        }),
    ],
});
