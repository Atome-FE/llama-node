import { exec } from "child_process";

const platforms = {
    "darwin-arm64": "aarch64-apple-darwin",
    "darwin-x64": "x86_64-apple-darwin",
    "linux-x64-gnu": "x86_64-unknown-linux-gnu",
    "win32-x64-msvc": "x86_64-pc-windows-msvc",
};

const compile = () => {
    Object.entries(platforms).forEach(([platform, target]) => {
        const buildProcess = exec(
            `napi build --platform --target ${target} --release`
        );
        buildProcess.stdout?.pipe(process.stdout);
        buildProcess.stderr?.pipe(process.stderr);
    });
};

compile();
