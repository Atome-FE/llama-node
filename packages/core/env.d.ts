/// <reference types="vite/client" />

interface ImportMetaEnv {
    model: string;
    // more env variables...
}

interface ImportMeta {
    readonly env: ImportMetaEnv;
}
