#!/usr/bin/env node
import { convert } from "@llama-node/core";
import yargs from "yargs";
import path from "path";
import { existsSync } from "fs";

const convertType = ["q4_0", "q4_1", "f16", "f32"] as const;

type ConvertType = typeof convertType[number];

class ConvertCommand implements yargs.CommandModule {
    command = "convert";
    describe = "Convert llama pth to ggml";
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
    async handler(args: yargs.Arguments) {
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

yargs
    .scriptName("llama")
    .usage("$0 <cmd> [args]")
    .command(new ConvertCommand())
    .demandCommand(1, "You need at least one command before moving on")
    .parse();
