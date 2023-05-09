#!/usr/bin/env node
import {
    convert,
    LLamaInferenceArguments,
    LLama,
    LLamaConfig,
    InferenceResultType,
} from "@llama-node/core";
import yargs from "yargs";
import path from "path";
import { existsSync } from "fs";

const convertType = ["q4_0", "q4_1", "f16", "f32"] as const;

type ConvertType = (typeof convertType)[number];

interface CLIInferenceArguments extends LLamaInferenceArguments, LLamaConfig {
    logger?: boolean;
}

class InferenceCommand implements yargs.CommandModule {
    command = "inference";
    describe = "Inference LLaMA";
    builder(args: yargs.Argv) {
        return (args as yargs.Argv<CLIInferenceArguments>)
            .help("help")
            .example('llama inference -p "How are you?"', "Inference LLaMA")
            .options("feedPrompt", {
                type: "boolean",
                demandOption: false,
                description: "Set it to true to hide promt feeding progress",
            })
            .options("float16", { type: "boolean", demandOption: false })
            .options("ignoreEos", { type: "boolean", demandOption: false })
            .options("nBatch", { type: "number", demandOption: false })
            .options("nThreads", { type: "number", demandOption: false })
            .options("numPredict", { type: "number", demandOption: false })
            .options("prompt", {
                type: "string",
                demandOption: true,
                alias: "p",
            })
            .options("repeatLastN", { type: "number", demandOption: false })
            .options("repeatPenalty", { type: "number", demandOption: false })
            .options("seed", { type: "number", demandOption: false })
            .options("temp", { type: "number", demandOption: false })
            .options("tokenBias", { type: "string", demandOption: false })
            .options("topK", { type: "number", demandOption: false })
            .options("topP", { type: "number", demandOption: false })
            .options("path", {
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
        const {
            $0,
            _,
            path: model,
            numCtxTokens,
            logger,
            ...rest
        } = args as yargs.ArgumentsCamelCase<CLIInferenceArguments>;
        const absolutePath = path.isAbsolute(model)
            ? model
            : path.join(process.cwd(), model);
        if (logger) {
            LLama.enableLogger();
        }
        const llama = await LLama.create({ path: absolutePath, numCtxTokens });
        llama.inference(rest, (result) => {
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

class ConvertCommand implements yargs.CommandModule<any, any> {
    command = "convert";
    describe = "Convert llama pth to ggml, not ready yet";
    builder(args: yargs.Argv) {
        return args
            .help("help")
            .example(
                "llama convert --dir ./model --type q4_0",
                "Convert pth to q4_0 ggml model"
            )
            .options({
                dir: {
                    describe: "The directory of model and tokenizer directory",
                    type: "string",
                    demandOption: true,
                },
            })
            .options({
                type: {
                    describe: "The type of model",
                    type: "string",
                    choices: convertType,
                    demandOption: true,
                },
            });
    }
    async handler(args: yargs.ArgumentsCamelCase) {
        const dir = args.dir as string;
        const type = args.type as ConvertType;

        const absolute = path.isAbsolute(dir)
            ? dir
            : path.join(process.cwd(), dir);
        if (!existsSync(absolute)) {
            console.error(`Directory ${absolute} does not exist`);
            return;
        } else {
            const elementType = convertType.findIndex((t) => t === type);
            await convert(absolute, elementType);
            console.log("Convert successfully");
        }
    }
}

(yargs as yargs.Argv<any | CLIInferenceArguments>)
    .scriptName("llama")
    .usage("$0 <cmd> [args]")
    .command(new ConvertCommand())
    .command(new InferenceCommand())
    .demandCommand(1, "You need at least one command before moving on")
    .strict()
    .parse();
