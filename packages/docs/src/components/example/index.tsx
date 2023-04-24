import React from "react";
import CodeBlock from "@theme/CodeBlock";

export const Source: React.FC<{ fileName?: string; name?: string }> = ({
    name,
    fileName,
}) => {
    const jsonFile: {
        name: string;
        markdown: string;
    }[] = require(`./${fileName}.json`);

    const example = jsonFile.find((block) => block.name === name);
    return <CodeBlock language="js">{example.markdown}</CodeBlock>;
};
