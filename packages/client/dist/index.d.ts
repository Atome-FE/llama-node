import { LLamaConfig, LLamaArguments } from '@llama-node/core';

declare class LLama {
    private llamaNode;
    constructor(config: LLamaConfig, enableLogger?: boolean);
    createCompletion: (params: LLamaArguments, callback: (data: {
        token: string;
        completed: boolean;
    }) => void) => Promise<boolean>;
}

export { LLama };
