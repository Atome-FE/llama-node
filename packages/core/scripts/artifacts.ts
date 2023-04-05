import axios from "axios";
import { execSync } from "child_process";
import glob from "glob";

async function getLatestGithubAction(repoOwner: string, repoName: string) {
    try {
        // Get the latest workflow runs for the specified repository
        const res = await axios.get(
            `https://api.github.com/repos/${repoOwner}/${repoName}/actions/runs`
        );

        const lastId: string = res.data.workflow_runs[0].id;

        return lastId;
    } catch (error) {
        console.error(error);
    }
}

const repoOwner = "hlhr202";
const repoName = "llama-node";

getLatestGithubAction(repoOwner, repoName).then(async (latestStep) => {
    execSync(`rimraf ${process.cwd()}/@llama-node`);
    execSync(`mkdir ${process.cwd()}/@llama-node`);
    execSync(`rimraf ${process.cwd()}/tmp/artifacts/`);
    execSync(
        `gh run download ${latestStep} --repo ${repoOwner}/${repoName} --dir ${process.cwd()}/tmp/artifacts/`
    );
    const files = await glob(`${process.cwd()}/tmp/artifacts/**/*.node`);
    files.forEach((file) => {
        execSync(`cp ${file} ${process.cwd()}/@llama-node`);
    });
    execSync(`rm -r ${process.cwd()}/tmp/artifacts/`);
});
