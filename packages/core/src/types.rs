use napi::bindgen_prelude::*;
use std::sync::mpsc::Sender;

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
 * Embedding result
 */
#[napi(string_enum)]
#[derive(Debug)]
pub enum EmbeddingResultType {
  Data,
  Error,
}

#[napi(object)]
#[derive(Clone, Debug)]
pub struct EmbeddingResult {
  pub r#type: EmbeddingResultType,
  pub message: Option<String>,
  pub data: Option<Vec<f64>>,
}

/**
 * Tokenize result
 */
#[napi(string_enum)]
#[derive(Debug)]
pub enum TokenizeResultType {
  Data,
}

#[napi(object)]
#[derive(Clone, Debug)]
pub struct TokenizeResult {
  pub r#type: TokenizeResultType,
  pub data: Vec<i32>,
}

/**
 * LLama model load config
 */
#[napi(object)]
#[derive(Clone, Debug)]
pub struct LLamaConfig {
  pub path: String,
  pub num_ctx_tokens: Option<i64>,
}

#[napi(object)]
#[derive(Clone, Debug)]
pub struct LoadModelResult {
  pub error: bool,
  pub message: Option<String>,
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
