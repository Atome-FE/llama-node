#![deny(clippy::all)]

#[macro_use]
extern crate napi_derive;

mod context;
mod llama;
mod tokenizer;
mod types;

use std::sync::Arc;

use llama::LLamaInternal;
use napi::{
    bindgen_prelude::*,
    threadsafe_function::{
        ErrorStrategy, ThreadSafeCallContext, ThreadsafeFunction, ThreadsafeFunctionCallMode,
    },
    JsFunction,
};
use tokio::sync::Mutex;
use types::{InferenceResult, LlamaContextParams, LlamaInvocation};

#[napi]
pub struct LLama {
    llama: Arc<Mutex<LLamaInternal>>,
}

#[napi]
impl LLama {
    #[napi]
    pub async fn load(
        path: String,
        params: Option<LlamaContextParams>,
        enable_logger: bool,
    ) -> Result<LLama> {
        if enable_logger {
            env_logger::builder()
                .filter_level(log::LevelFilter::Info)
                .parse_default_env()
                .init();
        }

        Ok(Self {
            llama: LLamaInternal::load(path, params, enable_logger).await,
        })
    }

    #[napi]
    pub async fn get_word_embedding(&self, params: LlamaInvocation) -> Result<Vec<f64>> {
        let llama = self.llama.lock().await;
        llama.embedding(&params).await
    }

    #[napi]
    pub async fn tokenize(&self, params: String, n_ctx: i32) -> Result<Vec<i32>> {
        let llama = self.llama.lock().await;
        llama.tokenize(&params, n_ctx as usize).await
    }

    #[napi]
    pub fn inference(
        &self,
        params: LlamaInvocation,
        #[napi(ts_arg_type = "(result: InferenceResult) => void")] callback: JsFunction,
    ) -> Result<()> {
        let tsfn: ThreadsafeFunction<InferenceResult, ErrorStrategy::Fatal> = callback
            .create_threadsafe_function(0, |ctx: ThreadSafeCallContext<InferenceResult>| {
                Ok(vec![ctx.value])
            })?;

        let llama = self.llama.clone();

        tokio::spawn(async move {
            let llama = llama.lock().await;
            llama
                .inference(&params, |result| {
                    tsfn.call(result, ThreadsafeFunctionCallMode::NonBlocking);
                })
                .await;
        });

        Ok(())
    }
}
