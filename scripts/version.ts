import main from "../package.json";
import cli from "../packages/cli/package.json";
import core from "../packages/core/package.json";
import cpp from "../packages/llama-cpp/package.json";
import rwkv from "../packages/rwkv-cpp/package.json";
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

const newVersionIsGreaterThanAll = [main, cli, example, core, cpp, rwkv].every(
    (pkg) => semver.gte(newVersion, pkg.version)
);

if (!newVersionIsGreaterThanAll) {
    console.error("New version must be greater than all existing versions");
    process.exit(1);
}

const backendPackages = { core, "llama-cpp": cpp, "rwkv-cpp": rwkv };

const updateVersionForBackend = (
    backend: "core" | "llama-cpp" | "rwkv-cpp"
) => {
    backendPackages[backend].version = newVersion;
    // write backend package.json
    fs.writeFileSync(
        path.join(__dirname, `../packages/${backend}/package.json`),
        JSON.stringify(backendPackages[backend], null, 2)
    );
    main.devDependencies[`@llama-node/${backend}`] = newVersion;
    main.optionalDependencies[`@llama-node/${backend}`] = newVersion;
    main.peerDependencies[`@llama-node/${backend}`] = newVersion;
    example.dependencies[`@llama-node/${backend}`] = newVersion;
};

console.log("Current versions: ");
console.log(`main: ${main.version}`);
console.log(`cli: ${cli.version}`);
console.log(`example: ${example.version}`);

console.log(`New version: ${newVersion}`);

console.log("Updating versions...");
main.version = newVersion;
cli.version = newVersion;
example.version = newVersion;
main.dependencies["@llama-node/cli"] = newVersion;
main.devDependencies["@llama-node/cli"] = newVersion;
main.peerDependencies["@llama-node/cli"] = newVersion;
cli.dependencies["@llama-node/core"] = newVersion;
example.dependencies["llama-node"] = newVersion;

updateVersionForBackend("core");
updateVersionForBackend("llama-cpp");
updateVersionForBackend("rwkv-cpp");

// write main package.json
fs.writeFileSync(
    path.join(__dirname, "../package.json"),
    JSON.stringify(main, null, 2)
);

// write cli package.json
fs.writeFileSync(
    path.join(__dirname, "../packages/cli/package.json"),
    JSON.stringify(cli, null, 2)
);

// write example package.json
fs.writeFileSync(
    path.join(__dirname, "../example/package.json"),
    JSON.stringify(example, null, 2)
);

console.log("Done!");
