import { Tokenizer } from "../nodejs/tokenizer-node.js";

import tokenizerJson from "../../../20B_tokenizer.json";

const tokenizer = new Tokenizer(JSON.stringify(tokenizerJson));

const text = "Hello, world!";

const tokens = tokenizer.encode(text, true);

console.log(tokens.ids);

tokens.free();
tokenizer.free();
