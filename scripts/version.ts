import main from "../package.json";
import core from "../packages/core/package.json";
import cpp from "../packages/llama-cpp/package.json";
import cli from "../packages/cli/package.json";
import example from "../example/package.json";
import semver from "semver";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const newVersion = process.argv[2];

if (!newVersion) {
    console.error("No version specified");
    process.exit(1);
}

if (!semver.valid(newVersion)) {
    console.error("Invalid version specified");
    process.exit(1);
}

const newVersionIsGreaterThanAll = [main, core, cli, example].every((pkg) => {
    return semver.gt(newVersion, pkg.version);
});

if (!newVersionIsGreaterThanAll) {
    console.error("New version must be greater than all existing versions");
    process.exit(1);
}

console.log("Current versions: ");
console.log(`main: ${main.version}`);
console.log(`core: ${core.version}`);
console.log(`cli: ${cli.version}`);
console.log(`cpp: ${cpp.version}`);
console.log(`example: ${example.version}`);

console.log(`New version: ${newVersion}`);

console.log("Updating versions...");
main.version = newVersion;
core.version = newVersion;
cli.version = newVersion;
cpp.version = newVersion;
example.version = newVersion;
main.dependencies["@llama-node/cli"] = newVersion;
main.optionalDependencies["@llama-node/core"] = newVersion;
main.optionalDependencies["@llama-node/llama-cpp"] = newVersion;
main.devDependencies["@llama-node/cli"] = newVersion;
main.devDependencies["@llama-node/core"] = newVersion;
main.devDependencies["@llama-node/llama-cpp"] = newVersion;
main.peerDependencies["@llama-node/core"] = newVersion;
main.peerDependencies["@llama-node/llama-cpp"] = newVersion;
main.peerDependencies["@llama-node/cli"] = newVersion;
cli.dependencies["@llama-node/core"] = newVersion;
example.dependencies["llama-node"] = newVersion;
example.dependencies["@llama-node/core"] = newVersion;
example.dependencies["@llama-node/llama-cpp"] = newVersion;

console.log("Writing new versions...");
fs.writeFileSync(
    path.join(__dirname, "../package.json"),
    JSON.stringify(main, null, 2)
);

fs.writeFileSync(
    path.join(__dirname, "../packages/core/package.json"),
    JSON.stringify(core, null, 2)
);

fs.writeFileSync(
    path.join(__dirname, "../packages/llama-cpp/package.json"),
    JSON.stringify(cpp, null, 2)
);

fs.writeFileSync(
    path.join(__dirname, "../packages/cli/package.json"),
    JSON.stringify(cli, null, 2)
);

fs.writeFileSync(
    path.join(__dirname, "../example/package.json"),
    JSON.stringify(example, null, 2)
);

console.log("Done!");
