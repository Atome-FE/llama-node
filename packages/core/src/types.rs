use llm::TokenId;
use napi::bindgen_prelude::*;
use serde::{Deserialize, Serialize};

#[napi(string_enum)]
#[derive(Debug)]
pub enum ModelType {
    Llama,
    Bloom,
    Gpt2,
    GptJ,
    GptNeoX,
    Mpt
}

#[napi(object)]
#[derive(Clone, Debug)]
pub struct InferenceToken {
    pub token: String,
    pub completed: bool,
}

#[napi(string_enum)]
#[derive(Debug)]
pub enum InferenceResultType {
    Data,
    End,
    Error,
}

#[napi(object)]
#[derive(Clone, Debug)]
pub struct InferenceResult {
    pub r#type: InferenceResultType,
    pub message: Option<String>,
    pub data: Option<InferenceToken>,
}

#[napi(object)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenBias {
    pub token_id: TokenId,
    pub bias: f64,
}

/**
 * LLM model load config
 */
#[napi(object)]
#[derive(Clone, Debug)]
pub struct ModelLoad {
    pub model_type: ModelType,

    /// Path of the model
    pub model_path: String,

    /// Sets the size of the context (in tokens). Allows feeding longer prompts.
    /// Note that this affects memory.
    ///
    /// LLaMA models are trained with a context size of 2048 tokens. If you
    /// want to use a larger context size, you will need to retrain the model,
    /// or use a model that was trained with a larger context size.
    ///
    /// Alternate methods to extend the context, including
    /// [context clearing](https://github.com/rustformers/llm/issues/77) are
    /// being investigated, but are not yet implemented. Additionally, these
    /// will likely not perform as well as a model with a larger context size.
    /// Default is 2048
    pub num_ctx_tokens: Option<i64>,

    /// MMapped files are faster, but may not work on all systems.
    /// Default is true
    pub use_mmap: Option<bool>,

    /// Path to the Lora file to apply to the model
    /// Default is None
    pub lora_path: Option<String>,
}

#[napi(object)]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Generate {
    /// Sets the number of threads to use
    /// Default is 4
    pub num_threads: i32,

    /// Number of tokens to predict
    /// Default is 512
    pub num_predict: i64,

    /// How many tokens from the prompt at a time to feed the network. Does not
    /// affect generation.
    /// Default is 8
    pub batch_size: i64,

    /// Size of the 'last N' buffer that is used for the `repeat_penalty`
    /// Default is 64
    pub repeat_last_n: i64,

    /// The penalty for repeating tokens. Higher values make the generation less
    /// likely to get into a loop, but may harm results when repetitive outputs
    /// are desired.
    /// Default is 1.30
    pub repeat_penalty: f64,

    /// Temperature, higher is more creative, should between 0 to 1
    /// Default is 0.80
    pub temperature: f64,

    /// Top-K: The top K words by score are kept during sampling.
    /// Default is 40
    pub top_k: i64,

    /// Top-p: The cumulative probability after which no more words are kept
    /// for sampling.
    /// Default is 0.95
    pub top_p: f64,

    /// Specifies the seed to use during sampling. Note that, depending on
    /// hardware, the same seed may lead to different results on two separate
    /// machines.
    /// Default is None
    pub seed: Option<i64>,

    /// Use 16-bit floats for model memory key and value. Ignored when restoring
    /// from the cache.
    /// Default is false
    pub float16: bool,

    /// Prompt for inference
    pub prompt: String,

    /// A comma separated list of token biases. The list should be in the format
    /// "TID=BIAS,TID=BIAS" where TID is an integer token ID and BIAS is a
    /// floating point number.
    /// For example, "1=-1.0,2=-1.0" sets the bias for token IDs 1
    /// (start of document) and 2 (end of document) to -1.0 which effectively
    /// disables the model from generating responses containing those token IDs.
    /// Default is None
    pub token_bias: Option<Vec<TokenBias>>,

    /// Prevent the end of stream (EOS/EOD) token from being generated. This will allow the
    /// model to generate text until it runs out of context space. Note: The --token-bias
    /// option will override this if specified.
    /// Default is false
    pub ignore_eos: bool,

    /// Feed prompt before inference, will hide feeded tokens in inference result
    /// Default is false
    pub feed_prompt: bool,

    /// Only feed prompt, will not execute inference
    /// When feed_prompt_only is true, feed_prompt will always be true
    /// Default is false
    pub feed_prompt_only: bool,

    /// Load session path
    /// Default is None
    pub load_session: Option<String>,

    /// Persist session path
    /// Default is None
    pub save_session: Option<String>,
}

impl Default for Generate {
    fn default() -> Self {
        Self {
            num_threads: num_cpus::get_physical() as i32,
            num_predict: 512,
            batch_size: 8,
            repeat_last_n: 64,
            repeat_penalty: 1.30,
            temperature: 0.80,
            top_k: 40,
            top_p: 0.95,
            seed: None,
            float16: false,
            prompt: "".to_string(),
            token_bias: None,
            ignore_eos: false,
            feed_prompt: false,
            feed_prompt_only: false,
            load_session: None,
            save_session: None,
        }
    }
}
