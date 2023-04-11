import axios from "axios";
import { execSync } from "child_process";
import glob from "glob";
import path from "path";

const repoOwner = "hlhr202";
const repoName = "llama-node";
const branch = "main";

async function getLatestGithubAction() {
    try {
        // Get the latest workflow runs for the specified repository
        const res = await axios.get(
            `https://api.github.com/repos/${repoOwner}/${repoName}/actions/runs?branch=${branch}`
        );

        const lastId: string = res.data.workflow_runs[0].id;

        return lastId;
    } catch (error) {
        console.error(error);
    }
}

const moveForPackages = async (packageName: string) => {
    const targetBinaryDir = path.resolve(
        process.cwd(),
        `./packages/${packageName}/@llama-node`
    );

    const sourceBinaries = await glob(
        `${process.cwd()}/tmp/artifacts/**/${packageName}/**/*.node`
    );

    execSync(`rimraf ${targetBinaryDir}`);
    execSync(`mkdir ${targetBinaryDir}`);

    sourceBinaries.forEach((file) => {
        execSync(`cp ${file} ${targetBinaryDir}`);
    });
};

getLatestGithubAction().then(async (latestStep) => {
    execSync(`rimraf ${process.cwd()}/tmp/artifacts/`);
    execSync(
        `gh run download ${latestStep} --repo ${repoOwner}/${repoName} --dir ${process.cwd()}/tmp/artifacts/`
    );

    await moveForPackages("core");
    await moveForPackages("llama-cpp");

    execSync(`rm -r ${process.cwd()}/tmp/artifacts/`);
});
