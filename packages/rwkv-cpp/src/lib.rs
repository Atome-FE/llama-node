#![deny(clippy::all)]

#[macro_use]
extern crate napi_derive;

mod context;
mod rwkv;
mod sampling;
mod types;

use std::sync::Arc;
use common_rs::logger::LLamaLogger;

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
        let logger = LLamaLogger::get_singleton();

        logger.set_enabled(enable_logger);

        Ok(Self {
            rwkv: RWKVInternal::load(model_path, tokenizer_path, n_threads, enable_logger).await,
        })
    }

    #[napi]
    pub async fn tokenize(&self, params: String) -> Result<Vec<i32>> {
        let rwkv = self.rwkv.lock().await;
        rwkv.tokenize(&params).await
    }

    #[napi(ts_return_type = "() => void")]
    pub fn inference(
        &self,
        env: Env,
        params: RWKVInvocation,
        #[napi(ts_arg_type = "(result: InferenceResult) => void")] callback: JsFunction,
    ) -> Result<JsFunction> {
        let tsfn: ThreadsafeFunction<InferenceResult, ErrorStrategy::Fatal> = callback
            .create_threadsafe_function(0, |ctx: ThreadSafeCallContext<InferenceResult>| {
                Ok(vec![ctx.value])
            })?;

        let rwkv = self.rwkv.clone();

        let running = Arc::new(Mutex::new(true));

        {
            let running = running.clone();
            tokio::task::spawn_blocking(move || {
                let mut rwkv = rwkv.blocking_lock();
                rwkv.inference(&params, running, |result| {
                    tsfn.call(result, ThreadsafeFunctionCallMode::NonBlocking);
                });
            });
        }

        env.create_function_from_closure("abort_inference", move |_| {
            let mut running = running.blocking_lock();
            *running = false;
            Ok(())
        })
    }
}
