use napi::bindgen_prelude::*;

#[napi(object)]
#[derive(Clone, Debug)]
pub struct InferenceToken {
    pub token: String,
    pub completed: bool,
}

#[napi(string_enum)]
pub enum InferenceResultType {
    Error,
    Data,
    End,
}

#[napi(object)]
pub struct InferenceResult {
    pub r#type: InferenceResultType,
    pub data: Option<InferenceToken>,
    pub message: Option<String>,
}

#[napi(object)]
#[derive(Debug, Clone)]
pub struct LlamaInvocation {
    pub n_threads: i32,
    pub n_tok_predict: i32,
    pub top_k: i32,                     // 40
    pub top_p: Option<f64>,             // default 0.95f, 1.0 = disabled
    pub tfs_z: Option<f64>,             // default 1.00f, 1.0 = disabled
    pub temp: Option<f64>,              // default 0.80f, 1.0 = disabled
    pub typical_p: Option<f64>,         // default 1.00f, 1.0 = disabled
    pub repeat_penalty: Option<f64>,    // default 1.10f, 1.0 = disabled
    pub repeat_last_n: Option<i32>, // default 64, last n tokens to penalize (0 = disable penalty, -1 = context size)
    pub frequency_penalty: Option<f64>, // default 0.00f, 1.0 = disabled
    pub presence_penalty: Option<f64>, // default 0.00f, 1.0 = disabled
    pub stop_sequence: Option<String>,
    pub penalize_nl: Option<bool>,
    pub prompt: String,
}

// Represents the configuration parameters for a LLamaContext.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct LlamaContextParams {
    pub n_ctx: i32,
    pub n_parts: i32,
    pub n_gpu_layers: i32,
    pub seed: i32,
    pub f16_kv: bool,
    pub logits_all: bool,
    pub vocab_only: bool,
    pub use_mlock: bool,
    pub embedding: bool,
    pub use_mmap: bool,
    pub lora: Option<LlamaLoraAdaptor>,
}

#[napi(object)]
#[derive(Debug, Clone)]
pub struct LlamaLoraAdaptor {
    pub lora_adapter: String,
    pub lora_base: Option<String>,
    pub n_threads: i32,
}
