import fs from "fs";
import path from "path";
import * as glob from "glob";

const examplePath = path.resolve(__dirname, "../../../example");

const writeExample = (variant: string) => {
    const json = glob
        .sync(path.resolve(examplePath, `js/${variant}/*.js`))
        .map((file) => {
            const fileName = path.basename(file);
            const fileNameWithoutExtension = fileName.split(".")[0];
            const fileContent = fs.readFileSync(file, "utf-8");

            const markdown = `// ${fileName}
${fileContent}`;

            return {
                name: fileNameWithoutExtension,
                markdown,
            };
        });

    const destination = path.resolve(__dirname, "../src/components/example");

    fs.writeFileSync(
        path.resolve(destination, `./${variant}.json`),
        JSON.stringify(json, null, 2)
    );
};

writeExample("llm-rs");
writeExample("llama-cpp");
writeExample("rwkv-cpp");