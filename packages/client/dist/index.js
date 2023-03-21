"use strict";
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

// src/index.ts
var src_exports = {};
__export(src_exports, {
  LLama: () => LLama
});
module.exports = __toCommonJS(src_exports);
var import_core = require("@llama-node/core");
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
      import_core.LLama.enableLogger();
    }
    this.llamaNode = import_core.LLama.create(config);
  }
};
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  LLama
});
//# sourceMappingURL=index.js.map