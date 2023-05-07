import init, { Tokenizer } from "../web";
import json from "../../../20B_tokenizer.json";

await init();

const tokenizer = new Tokenizer(JSON.stringify(json));

const text = "Hello, world!";

const tokens = tokenizer.encode(text, true);

console.log(tokens.ids);

tokens.free();

tokenizer.free();
