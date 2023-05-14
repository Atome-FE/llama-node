import axios from "axios";
import { execSync } from "child_process";
import glob from "glob";
import path from "path";

const repoOwner = "Atome-FE";
const repoName = "llama-node";
const branch = process.argv[2] ?? "main";

interface WorkflowRun {
    id: string;
    status: "queued" | "in_progress" | "completed";
    display_title?: string;
    conclusion:
        | "success"
        | "failure"
        | "cancelled"
        | "skipped"
        | "timed_out"
        | "action_required"
        | "stale"
        | null;
}

async function getLatestGithubAction() {
    try {
        // Get the latest workflow runs for the specified repository
        const res = await axios.get(
            `https://api.github.com/repos/${repoOwner}/${repoName}/actions/runs?branch=${branch}`
        );

        const workflowRuns: WorkflowRun[] = res.data.workflow_runs;

        const firstThatSuccess = workflowRuns.find(
            (run) => run.status === "completed" && run.conclusion === "success"
        );

        const lastId = firstThatSuccess?.id;

        console.log(`Branch: ${branch}`);
        console.log(`Last successful workflow run id: ${lastId}`);
        console.log(
            `Last successful workflow description: ${firstThatSuccess?.display_title}`
        );

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
        `${process.cwd()}/tmp/artifacts/*/${packageName}/**/*.node`
    );

    execSync(`rimraf ${targetBinaryDir}`);
    execSync(`mkdir ${targetBinaryDir}`);

    sourceBinaries.forEach((file) => {
        execSync(`cp ${file} ${targetBinaryDir}`);
        const fileName = path.basename(file);
        console.log(`Copied ${fileName} to ${targetBinaryDir}`);
    });
};

getLatestGithubAction().then(async (latestStep) => {
    console.log("Downloading artifacts...");
    execSync(`rimraf ${process.cwd()}/tmp/artifacts/`);
    execSync(
        `gh run download ${latestStep} --repo ${repoOwner}/${repoName} --dir ${process.cwd()}/tmp/artifacts/`,
        { stdio: "inherit" }
    );

    await moveForPackages("core");
    await moveForPackages("llama-cpp");
    await moveForPackages("rwkv-cpp");

    execSync(`rm -r ${process.cwd()}/tmp/artifacts/`);
    console.log("Done!");
});
