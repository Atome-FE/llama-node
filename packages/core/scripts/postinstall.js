const { execSync } = require("child_process");
const { readFileSync, existsSync } = require("fs");
const { join } = require("path");

const { platform, arch } = process;

let localFileExisted = false;

function isMusl() {
    // For Node 10
    if (!process.report || typeof process.report.getReport !== "function") {
        try {
            const lddPath = require("child_process")
                .execSync("which ldd")
                .toString()
                .trim();
            return readFileSync(lddPath, "utf8").includes("musl");
        } catch (e) {
            return true;
        }
    } else {
        const { glibcVersionRuntime } = process.report.getReport().header;
        return !glibcVersionRuntime;
    }
}

switch (platform) {
    case "android":
        switch (arch) {
            case "arm64":
                localFileExisted = existsSync(
                    join(__dirname, "../@llama-node/core.android-arm64.node")
                );
                break;
            case "arm":
                localFileExisted = existsSync(
                    join(__dirname, "../@llama-node/core.android-arm-eabi.node")
                );
                break;
            default:
                throw new Error(`Unsupported architecture on Android ${arch}`);
        }
        break;
    case "win32":
        switch (arch) {
            case "x64":
                localFileExisted = existsSync(
                    join(__dirname, "../@llama-node/core.win32-x64-msvc.node")
                );
                break;
            case "ia32":
                localFileExisted = existsSync(
                    join(__dirname, "../@llama-node/core.win32-ia32-msvc.node")
                );
                break;
            case "arm64":
                localFileExisted = existsSync(
                    join(__dirname, "../@llama-node/core.win32-arm64-msvc.node")
                );
                break;
            default:
                throw new Error(`Unsupported architecture on Windows: ${arch}`);
        }
        break;
    case "darwin":
        localFileExisted = existsSync(
            join(__dirname, "../@llama-node/core.darwin-universal.node")
        );

        switch (arch) {
            case "x64":
                localFileExisted = existsSync(
                    join(__dirname, "../@llama-node/core.darwin-x64.node")
                );
                break;
            case "arm64":
                localFileExisted = existsSync(
                    join(__dirname, "../@llama-node/core.darwin-arm64.node")
                );
                break;
            default:
                throw new Error(`Unsupported architecture on macOS: ${arch}`);
        }
        break;
    case "freebsd":
        if (arch !== "x64") {
            throw new Error(`Unsupported architecture on FreeBSD: ${arch}`);
        }
        localFileExisted = existsSync(
            join(__dirname, "../@llama-node/core.freebsd-x64.node")
        );

        break;
    case "linux":
        switch (arch) {
            case "x64":
                if (isMusl()) {
                    localFileExisted = existsSync(
                        join(
                            __dirname,
                            "../@llama-node/core.linux-x64-musl.node"
                        )
                    );
                } else {
                    localFileExisted = existsSync(
                        join(
                            __dirname,
                            "../@llama-node/core.linux-x64-gnu.node"
                        )
                    );
                }
                break;
            case "arm64":
                if (isMusl()) {
                    localFileExisted = existsSync(
                        join(
                            __dirname,
                            "../@llama-node/core.linux-arm64-musl.node"
                        )
                    );
                } else {
                    localFileExisted = existsSync(
                        join(
                            __dirname,
                            "../@llama-node/core.linux-arm64-gnu.node"
                        )
                    );
                }
                break;
            case "arm":
                localFileExisted = existsSync(
                    join(
                        __dirname,
                        "../@llama-node/core.linux-arm-gnueabihf.node"
                    )
                );
                break;
            default:
                throw new Error(`Unsupported architecture on Linux: ${arch}`);
        }
        break;
    default:
        throw new Error(`Unsupported OS: ${platform}, architecture: ${arch}`);
}

if (!localFileExisted) {
    console.log("Building native module...");
    execSync("npm run build");
} else {
    console.log("Skipping build, local file already exists.");
}
