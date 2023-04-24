---
sidebar_position: 2
---

# 模型版本

llama.cpp 社区有多个历史模型版本. 请注意特定的推理后端只支持特定的模型版本.

## llama.cpp

以下是llama.cpp支持的模型类型，ggml.h源码中可找到：

```c
enum ggml_type {
    // explicitly numbered values are used in llama.cpp files
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q4_2 = 4,
    GGML_TYPE_Q4_3 = 5,
    GGML_TYPE_Q8_0 = 6,
    GGML_TYPE_I8,
    GGML_TYPE_I16,
    GGML_TYPE_I32,
    GGML_TYPE_COUNT,
};
```

---

## llama-rs

以下是llama-rs支持的模型类型，从llama-rs的ggml绑定中可找到：

```rust
pub enum Type {
    /// Quantized 4-bit (type 0).
    #[default]
    Q4_0,
    /// Quantized 4-bit (type 1); used by GPTQ.
    Q4_1,
    /// Integer 32-bit.
    I32,
    /// Float 16-bit.
    F16,
    /// Float 32-bit.
    F32,
}
```

llama-rs也支持旧版的ggml/ggmf模型