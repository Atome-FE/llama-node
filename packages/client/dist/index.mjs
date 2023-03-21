// src/index.ts
import {
  LLama as LLamaNode
} from "@llama-node/core";
var LLama = class {
  constructor(config, enableLogger) {
    this.createCompletion = (params, callback) => {
      let completed = false;
      return new Promise((res, rej) => {
        this.llamaNode.inference(params, (response) => {
          switch (response.type) {
            case "DATA": {
              const data = {
                token: response.data.token,
                completed: !!response.data.completed
              };
              if (data.completed) {
                completed = true;
              }
              callback(data);
              break;
            }
            case "END": {
              res(completed);
              break;
            }
            case "ERROR": {
              rej(response.message);
              break;
            }
          }
        });
      });
    };
    if (enableLogger) {
      LLamaNode.enableLogger();
    }
    this.llamaNode = LLamaNode.create(config);
  }
};
export {
  LLama
};
//# sourceMappingURL=index.mjs.map