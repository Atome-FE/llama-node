import { defineConfig } from "tsup";

export default defineConfig({
    entry: ["src/index.ts"],
    target: ["es2015"],
    format: ["cjs", "esm"],
    dts: true,
    shims: true,
    splitting: false,
    sourcemap: true,
    clean: true,
});
