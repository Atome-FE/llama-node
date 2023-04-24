---
sidebar_position: 3
---

# Backends

llama-node currently supports llama-rs and llama.cpp backends.

The current version supports only one inference session on one LLama instance at the same time

If you wish to have multiple inference sessions concurrently, you need to create multiple LLama instances

---

- To use llama.cpp backend, run

```bash
npm install @llama-node/llama-cpp
```

- To use llama-rs backend, run

```bash
npm install @llama-node/core
```