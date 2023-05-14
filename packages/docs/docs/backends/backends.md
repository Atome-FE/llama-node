---
sidebar_position: 3
---

# Backends

llama-node currently supports llm-rs, llama.cpp and rwkv.cpp backends.

llm-rs can supported multiple inference at same time.

llama.cpp and rwkv.cpp will treat async inference (in concurrent) as sequential requests.

---

- To use llama.cpp backend, run

```bash
npm install @llama-node/llama-cpp
```

- To use llm-rs backend, run

```bash
npm install @llama-node/core
```

- To use rwkv.cpp backend, run

```bash
npm install @llama-node/rwkv-cpp
```