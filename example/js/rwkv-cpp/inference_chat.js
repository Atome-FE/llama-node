import { LLM } from "llama-node";
import { RwkvCpp } from "llama-node/dist/llm/rwkv-cpp.js";
import path from "path";
const modelPath = path.resolve(process.cwd(), "../ggml-rwkv-4_raven-7b-v9-Eng99%-20230412-ctx8192-Q4_1_0.bin");
const tokenizerPath = path.resolve(process.cwd(), "../20B_tokenizer.json");
const rwkv = new LLM(RwkvCpp);
const config = {
    modelPath,
    tokenizerPath,
    nThreads: 4,
    enableLogging: true,
};
const prompt = `The following is a coherent verbose detailed conversation between a girl named Alice and her friend Bob. Alice is very intelligent, creative and friendly. Alice is unlikely to disagree with Bob, and Alice doesn't like to ask Bob questions. Alice likes to tell Bob a lot about herself and her opinions. Alice usually gives Bob kind, helpful and informative advices.\n\nBob: Hello Alice, how are you doing?\n\nAlice: Hi! Thanks, I'm fine. What about you?\n\nBob: I am fine. It's nice to see you. Look, here is a store selling tea and juice.\n\nAlice: Sure. Let's go inside. I would like to have some Mocha latte, which is my favourite!\n\nBob: What is it?\n\nAlice: Mocha latte is usually made with espresso, milk, chocolate, and frothed milk. Its flavors are frequently sweet.\n\nBob: Sounds tasty. I'll try it next time. Would you like to chat with me for a while?\n\nAlice: Of course! I'm glad to answer your questions or give helpful advices. You know, I am confident with my expertise. So please go ahead!\n\n`;
const run = async () => {
    await rwkv.load(config);
    // init session data
    const params = {
        maxPredictLength: 2048,
        topP: 0.1,
        temp: 0.1,
        prompt,
        sessionFilePath: path.resolve(process.cwd(), "../../session1.bin"),
        isSkipGeneration: true,
        isOverwriteSessionFile: true
    };
    await rwkv.createCompletion(params, (response) => {
        process.stdout.write(response.token);
    });
    // reuse session data, you don't need process all prompt once the session already initialized
    const params2 = {
        maxPredictLength: 2048,
        // For better Q&A accuracy and less diversity, reduce top_p (to 0.5, 0.2, 0.1 etc.)
        topP: 0.1,
        // Sampling temperature. It could be a good idea to increase temperature when top_p is low.
        temp: 0.1,
        prompt: 'Bob: Who are you?\\n\\nAlice: ',
        endString: '\n\n',
        sessionFilePath: path.resolve(process.cwd(), "../../session1.bin"),
        // set to false will keep the initial state of session
        isOverwriteSessionFile: true,
        // Penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
        presencePenalty: 0.2,
        // Penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
        frequencyPenalty: 0.2
    };
    await rwkv.createCompletion(params2, (response) => {
        process.stdout.write(response.token);
    });
};
run();
