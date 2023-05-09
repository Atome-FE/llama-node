#![deny(clippy::all)]

#[macro_use]
extern crate napi_derive;

mod context;
mod rwkv;
mod sampling;
mod types;

use std::sync::Arc;

use context::RWKVInvocation;
use napi::{
    bindgen_prelude::*,
    threadsafe_function::{
        ErrorStrategy, ThreadSafeCallContext, ThreadsafeFunction, ThreadsafeFunctionCallMode,
    },
    JsFunction,
};
use rwkv::RWKVInternal;
use tokio::sync::Mutex;
use types::InferenceResult;

#[napi]
pub struct RWKV {
    rwkv: Arc<Mutex<RWKVInternal>>,
}

#[napi]
impl RWKV {
    #[napi]
    pub async fn load(
        model_path: String,
        tokenizer_path: String,
        n_threads: u32,
        enable_logger: bool,
    ) -> Result<RWKV> {
        if enable_logger {
            env_logger::builder()
                .filter_level(log::LevelFilter::Info)
                .parse_default_env()
                .init();
        }

        Ok(Self {
            rwkv: RWKVInternal::load(model_path, tokenizer_path, n_threads, enable_logger).await,
        })
    }

    #[napi]
    pub async fn tokenize(&self, params: String) -> Result<Vec<i32>> {
        let rwkv = self.rwkv.lock().await;
        rwkv.tokenize(&params).await
    }

    #[napi]
    pub fn inference(
        &self,
        params: RWKVInvocation,
        #[napi(ts_arg_type = "(result: InferenceResult) => void")] callback: JsFunction,
    ) -> Result<()> {
        let tsfn: ThreadsafeFunction<InferenceResult, ErrorStrategy::Fatal> = callback
            .create_threadsafe_function(0, |ctx: ThreadSafeCallContext<InferenceResult>| {
                Ok(vec![ctx.value])
            })?;

        let rwkv = self.rwkv.clone();
        tokio::spawn(async move {
            let mut rwkv = rwkv.lock().await;
            rwkv.inference(&params, |result| {
                tsfn.call(result, ThreadsafeFunctionCallMode::NonBlocking);
            })
            .await;
        });

        Ok(())
    }
}
