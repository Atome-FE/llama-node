#![deny(clippy::all)]

#[macro_use]
extern crate napi_derive;

mod context;
mod llama;
mod tokenizer;
mod types;

use std::sync::Arc;

use common_rs::logger::LLamaLogger;

use llama::LLamaInternal;
use napi::{
    bindgen_prelude::*,
    threadsafe_function::{
        ErrorStrategy, ThreadSafeCallContext, ThreadsafeFunction, ThreadsafeFunctionCallMode,
    },
    JsFunction,
};
use tokio::sync::Mutex;
use types::{InferenceResult, InferenceResultType, Generate, ModelLoad};

#[napi]
pub struct LLama {
    llama: Arc<Mutex<LLamaInternal>>,
}

#[napi]
impl LLama {
    #[napi]
    pub async fn load(
        #[napi(ts_arg_type = "Partial<LoadModel>")] params: serde_json::Value,
        enable_logger: bool,
    ) -> Result<LLama> {
        let params = serde_json::from_value::<ModelLoad>(params).unwrap();

        let logger = LLamaLogger::get_singleton();
        logger.set_enabled(enable_logger);

        Ok(Self {
            llama: LLamaInternal::load(params, enable_logger).await?,
        })
    }

    #[napi]
    pub async fn get_word_embedding(&self, params: Generate) -> Result<Vec<f64>> {
        let llama = self.llama.lock().await;
        llama.embedding(&params).await
    }

    #[napi]
    pub async fn tokenize(&self, params: String) -> Result<Vec<i32>> {
        let llama = self.llama.lock().await;
        llama.tokenize(&params).await
    }

    #[napi(ts_return_type = "() => void")]
    pub fn inference(
        &self,
        env: Env,
        params: Generate,
        #[napi(ts_arg_type = "(result: InferenceResult) => void")] callback: JsFunction,
    ) -> Result<JsFunction> {
        let tsfn: ThreadsafeFunction<InferenceResult, ErrorStrategy::Fatal> = callback
            .create_threadsafe_function(0, |ctx: ThreadSafeCallContext<InferenceResult>| {
                Ok(vec![ctx.value])
            })?;

        let llama = self.llama.clone();

        let running = Arc::new(Mutex::new(true));

        {
            let running = running.clone();
            tokio::task::spawn_blocking(move || {
                let llama = llama.blocking_lock();
                let res = llama.inference(&params, running, |result| {
                    tsfn.call(result, ThreadsafeFunctionCallMode::NonBlocking);
                });
                if let Err(e) = res {
                    tsfn.call(
                        InferenceResult {
                            r#type: InferenceResultType::Error,
                            data: None,
                            message: Some(format!("Failed to run inference: {:?}", e)),
                        },
                        ThreadsafeFunctionCallMode::NonBlocking,
                    );
                }
            });
        }

        env.create_function_from_closure("abort_inference", move |_| {
            let mut running = running.blocking_lock();
            *running = false;
            Ok(())
        })
    }
}
