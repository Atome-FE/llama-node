use napi::bindgen_prelude::*;

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

/**
 * LLama model load config
 */
#[napi(object)]
#[derive(Clone, Debug)]
pub struct LLamaConfig {
    /// Path of the model
    pub path: String,

    /// Sets the size of the context (in tokens). Allows feeding longer prompts.
    /// Note that this affects memory.
    ///
    /// LLaMA models are trained with a context size of 2048 tokens. If you
    /// want to use a larger context size, you will need to retrain the model,
    /// or use a model that was trained with a larger context size.
    ///
    /// Alternate methods to extend the context, including
    /// [context clearing](https://github.com/rustformers/llama-rs/issues/77) are
    /// being investigated, but are not yet implemented. Additionally, these
    /// will likely not perform as well as a model with a larger context size.
    /// Default is 512
    pub num_ctx_tokens: Option<i64>,

    /// MMapped files are faster, but may not work on all systems.
    /// Default is true
    pub use_mmap: Option<bool>,
}

#[napi(object)]
#[derive(Clone, Debug)]
pub struct LLamaInferenceArguments {
    /// Sets the number of threads to use
    /// Default is 4
    pub n_threads: Option<i32>,

    /// How many tokens from the prompt at a time to feed the network. Does not
    /// affect generation.
    /// Default is 8
    pub n_batch: Option<i64>,

    /// Top-K: The top K words by score are kept during sampling.
    /// Default is 30
    pub top_k: Option<i64>,

    /// Top-p: The cumulative probability after which no more words are kept
    /// for sampling.
    /// Default is 0.95
    pub top_p: Option<f64>,

    /// The penalty for repeating tokens. Higher values make the generation less
    /// likely to get into a loop, but may harm results when repetitive outputs
    /// are desired.
    /// Default is 1.30
    pub repeat_penalty: Option<f64>,

    /// Temperature, higher is more creative, should between 0 to 1
    /// Default is 0.8
    pub temp: Option<f64>,

    /// Specifies the seed to use during sampling. Note that, depending on
    /// hardware, the same seed may lead to different results on two separate
    /// machines.
    /// Default is None
    pub seed: Option<i64>,

    /// Number of tokens to predict
    /// Default is 512
    pub num_predict: Option<i64>,

    /// Size of the 'last N' buffer that is used for the `repeat_penalty`
    /// Default is 512
    pub repeat_last_n: Option<i64>,

    /// Prompt for inference
    pub prompt: String,

    /// Use 16-bit floats for model memory key and value. Ignored when restoring
    /// from the cache.
    /// Default is false
    pub float16: Option<bool>,

    /// A comma separated list of token biases. The list should be in the format
    /// "TID=BIAS,TID=BIAS" where TID is an integer token ID and BIAS is a
    /// floating point number.
    /// For example, "1=-1.0,2=-1.0" sets the bias for token IDs 1
    /// (start of document) and 2 (end of document) to -1.0 which effectively
    /// disables the model from generating responses containing those token IDs.
    /// Default is None
    pub token_bias: Option<String>,

    /// Prevent the end of stream (EOS/EOD) token from being generated. This will allow the
    /// model to generate text until it runs out of context space. Note: The --token-bias
    /// option will override this if specified.
    /// Default is false
    pub ignore_eos: Option<bool>,

    /// Feed prompt before inference, will hide feeded tokens in inference result
    /// Default is false
    pub feed_prompt: Option<bool>,

    /// Only feed prompt, will not execute inference
    /// When feed_prompt_only is true, feed_prompt will always be true
    /// Default is false
    pub feed_prompt_only: Option<bool>,

    /// Load session path
    /// Default is None
    pub load_session: Option<String>,

    /// Persist session path
    /// Default is None
    pub save_session: Option<String>,
}
