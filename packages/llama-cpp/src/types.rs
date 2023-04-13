use crate::context::LlamaInvocation;
use napi::bindgen_prelude::*;
use std::sync::mpsc::Sender;

#[derive(Clone, Debug)]
pub enum LLamaCommand {
    Inference(LlamaInvocation, Sender<InferenceResult>),
    Tokenize(String, usize, Sender<TokenizeResult>),
    Embedding(LlamaInvocation, Sender<EmbeddingResult>),
}

#[napi(string_enum)]
pub enum TokenizeResultType {
    Error,
    Data,
}

#[napi(object)]
pub struct TokenizeResult {
    pub r#type: TokenizeResultType,
    pub data: Vec<i32>,
}

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

#[napi(string_enum)]
pub enum EmbeddingResultType {
    Error,
    Data,
}

#[napi(object)]
pub struct EmbeddingResult {
    pub r#type: EmbeddingResultType,
    pub data: Vec<f64>,
}
