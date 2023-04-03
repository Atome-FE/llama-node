use std::sync::mpsc::Sender;

#[napi(object)]
#[derive(Clone, Debug)]
pub struct InferenceToken {
  pub token: String,
  pub completed: bool,
}

#[derive(Clone, Debug)]
pub enum InferenceResult {
  InferenceData(InferenceToken),
  InferenceError(String),
  InferenceEnd,
}

#[derive(Clone, Debug)]
pub enum EmbeddingResult {
  EmbeddingError(String),
  EmbeddingData(Option<Vec<f32>>)
}

#[napi(object)]
#[derive(Clone, Debug)]
pub struct LoadModelResult {
  pub error: bool,
  pub message: Option<String>,
}

#[napi(object)]
#[derive(Clone, Debug)]
pub struct TokenizeResult {
  pub data: Vec<i32>
}

#[napi(object)]
#[derive(Clone, Debug)]
pub struct LLamaConfig {
  pub path: String,
  pub num_ctx_tokens: Option<i64>,
}

#[napi(object)]
#[derive(Clone, Debug)]
pub struct LLamaInferenceArguments {
  pub n_threads: Option<i32>,
  pub n_batch: Option<i64>,
  pub top_k: Option<i64>,
  pub top_p: Option<f64>,
  pub repeat_penalty: Option<f64>,
  pub temp: Option<f64>,
  pub seed: Option<i64>,
  pub num_predict: Option<i64>,
  pub repeat_last_n: Option<i64>,
  pub prompt: String,
  pub float16: Option<bool>,
  pub token_bias: Option<String>,
  pub ignore_eos: Option<bool>,
  pub feed_prompt: Option<bool>,
}

#[derive(Clone, Debug)]
pub enum LLamaCommand {
  LoadModel(LLamaConfig, Sender<LoadModelResult>),
  Inference(LLamaInferenceArguments, Sender<InferenceResult>),
  Embedding(LLamaInferenceArguments, Sender<EmbeddingResult>),
  Tokenize(String, Sender<TokenizeResult>),
}