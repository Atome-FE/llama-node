import { exec } from "child_process";
import pAll from "p-all";

const getTargets = () => {
    switch (process.platform) {
        case "darwin": {
            return ["aarch64-apple-darwin", "x86_64-apple-darwin"];
        }
        case "linux": {
            return ["x86_64-unknown-linux-gnu", "x86_64-unknown-linux-musl"];
        }
        case "win32": {
            return ["x86_64-pc-windows-msvc"];
        }
        default:
            return [];
    }
};

const compile = async () => {
    const targets = getTargets();
    const promises = targets.map((target) => {
        const buildProcess = exec(
            `rustup target add ${target} && napi build --platform --target ${target} --release`
        );
        buildProcess.stdout?.pipe(process.stdout);
        buildProcess.stderr?.pipe(process.stderr);

        return () =>
            new Promise<boolean>((resolve, reject) => {
                buildProcess.on("close", (code) => {
                    if (code !== 0) {
                        reject(code);
                    } else {
                        resolve(true);
                    }
                });
            });
    });

    try {
        await pAll(promises, { concurrency: 1 });
    } catch (error) {
        console.error(error);
        process.exit(1);
    }
};

compile();
