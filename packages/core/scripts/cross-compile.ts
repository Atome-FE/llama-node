import { exec } from "child_process";

const platforms = {
    "darwin-arm64": "aarch64-apple-darwin",
    "darwin-x64": "x86_64-apple-darwin",
    "linux-x64-gnu": "x86_64-unknown-linux-gnu",
};

const compile = () => {
    Object.entries(platforms).forEach(([platform, target]) => {
        exec(
            `napi build --platform --target ${target} --release --zig`
        ).stdout?.pipe(process.stdout);
    });
};

compile();
