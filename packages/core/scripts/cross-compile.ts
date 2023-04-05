import { exec } from "child_process";

const getTargets = () => {
    switch (process.platform) {
        case "darwin": {
            return ["aarch64-apple-darwin", "x86_64-apple-darwin"];
        }
        case "linux": {
            return ["x86_64-unknown-linux-gnu"];
        }
        case "win32": {
            return ["x86_64-pc-windows-msvc"];
        }
        default:
            return [];
    }
};

const compile = () => {
    const targets = getTargets();
    targets.forEach((target) => {
        const buildProcess = exec(
            `napi build --platform --target ${target} --release`
        );
        buildProcess.stdout?.pipe(process.stdout);
        buildProcess.stderr?.pipe(process.stderr);
    });
};

compile();
