use std::sync::mpsc::Sender;
use napi::bindgen_prelude::{BigInt};

#[napi(object)]
#[derive(Clone, Debug)]
pub struct InferenceToken {
  pub token: String,
  pub completed: bool,
}

#[derive(Clone, Debug)]
pub enum InferenceResult {
  InferenceData(InferenceToken),
  InferenceEnd(Option<String>),
}

#[napi(object)]
#[derive(Clone, Debug)]
pub struct LoadModelResult {
  pub error: bool,
  pub message: Option<String>,
}

#[napi(object)]
#[derive(Clone, Debug)]
pub struct LLamaConfig {
  pub path: String,
  pub num_ctx_tokens: Option<i32>,
}

#[napi(object)]
#[derive(Clone, Debug)]
pub struct LLamaArguments {
  pub n_threads: Option<i32>,
  pub n_batch: Option<BigInt>,
  pub top_k: Option<BigInt>,
  pub top_p: Option<f64>,
  pub repeat_penalty: Option<f64>,
  pub temp: Option<f64>,
  pub seed: Option<BigInt>,
  pub num_predict: Option<BigInt>,
  pub repeat_last_n: Option<BigInt>,
  pub prompt: String,
  pub float16: Option<bool>,
  pub token_bias: Option<String>,
  pub ignore_eos: Option<bool>,
}

#[derive(Clone, Debug)]
pub enum LLamaCommand {
  LoadModel(LLamaConfig, Sender<LoadModelResult>),
  Inference(LLamaArguments, Sender<InferenceResult>),
}