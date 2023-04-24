---
sidebar_position: 2
---

# Model versioning

llama.cpp community has developed several versions of model. Please be careful that some model version is not supported by some of the backends.

## llama.cpp

For llama.cpp, supported model types can check from ggml.h source:

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

## llama-rs

For llama-rs, supported model types can check from llama-rs ggml bindings:

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

llama-rs also supports legacy llama.cpp models