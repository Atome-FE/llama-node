#!/usr/bin/env node
import {
    convert,
    Generate,
    Llm,
    ModelLoad,
    InferenceResultType,
} from "@llama-node/core";
import yargs from "yargs";
import path from "path";
import { existsSync } from "fs";

const convertType = ["q4_0", "q4_1", "f16", "f32"] as const;

type ConvertType = (typeof convertType)[number];

interface CLIInferenceArguments extends Partial<Generate>, ModelLoad {
    logger?: boolean;
}

class InferenceCommand implements yargs.CommandModule {
    command = "inference";
    describe = "Inference LLaMA";
    builder(args: yargs.Argv) {
        return (args as yargs.Argv<CLIInferenceArguments>)
            .help("help")
            .example('llama inference -p "How are you?"', "Inference LLaMA")
            .options("modelType", {
                type: "string",
                demandOption: true,
            })
            .options("feedPrompt", {
                type: "boolean",
                demandOption: false,
                description: "Set it to true to hide promt feeding progress",
            })
            .options("float16", { type: "boolean", demandOption: false })
            .options("ignoreEos", { type: "boolean", demandOption: false })
            .options("batchSize", { type: "number", demandOption: false })
            .options("numThreads", { type: "number", demandOption: false })
            .options("numPredict", { type: "number", demandOption: false })
            .options("prompt", {
                type: "string",
                demandOption: true,
                alias: "p",
            })
            .options("repeatLastN", { type: "number", demandOption: false })
            .options("repeatPenalty", { type: "number", demandOption: false })
            .options("seed", { type: "number", demandOption: false })
            .options("temperature", { type: "number", demandOption: false })
            .options("tokenBias", { type: "string", demandOption: false })
            .options("topK", { type: "number", demandOption: false })
            .options("topP", { type: "number", demandOption: false })
            .options("modelPath", {
                type: "string",
                demandOption: true,
                alias: ["m", "model"],
            })
            .options("numCtxTokens", { type: "number", demandOption: false })
            .options("logger", {
                type: "boolean",
                demandOption: false,
                default: true,
                alias: "verbose",
            });
    }
    async handler(args: yargs.ArgumentsCamelCase) {
        const { $0, _, modelPath, modelType, numCtxTokens, logger, ...rest } =
            args as yargs.ArgumentsCamelCase<CLIInferenceArguments>;
        const absolutePath = path.isAbsolute(modelPath)
            ? modelPath
            : path.join(process.cwd(), modelPath);
        const llm = await Llm.load(
            {
                modelPath: absolutePath,
                modelType,
                numCtxTokens,
            },
            logger ?? true
        );
        llm.inference(rest, (result) => {
            switch (result.type) {
                case InferenceResultType.Data:
                    process.stdout.write(result.data?.token ?? "");
                    break;
                case InferenceResultType.Error:
                    console.error(result.message);
                    break;
                case InferenceResultType.End:
                    break;
            }
        });
    }
}

(yargs as yargs.Argv<any | CLIInferenceArguments>)
    .scriptName("llama")
    .usage("$0 <cmd> [args]")
    .command(new InferenceCommand())
    .demandCommand(1, "You need at least one command before moving on")
    .strict()
    .parse();
