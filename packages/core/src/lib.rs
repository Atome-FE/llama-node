#![deny(clippy::all)]
#![allow(clippy::enum_variant_names)]

#[macro_use]
extern crate napi_derive;

mod context;
mod load;
mod types;

use std::{path::Path, sync::Arc};

use context::LLMContext;
use llm::InferenceFeedback;
use tokio::sync::Mutex;
use types::{Generate, InferenceResult, ModelLoad};

use napi::{
    bindgen_prelude::*,
    threadsafe_function::{
        ErrorStrategy, ThreadSafeCallContext, ThreadsafeFunction, ThreadsafeFunctionCallMode,
    },
    JsFunction,
};

use common_rs::logger::LLamaLogger;

#[napi]
pub enum ElementType {
    /// All tensors are stored as f32.
    F32,
    /// All tensors are mostly stored as `f16`, except for the 1D tensors (32-bit).
    MostlyF16,
    /// All tensors are mostly stored as `Q4_0`, except for the 1D tensors (32-bit).
    MostlyQ4_0,
    /// All tensors are mostly stored as `Q4_1`, except for the 1D tensors (32-bit)
    MostlyQ4_1,
    /// All tensors are mostly stored as `Q4_1`, except for the 1D tensors (32-bit)
    /// and the `tok_embeddings.weight` (f16) and `output.weight` tensors (f16).
    MostlyQ4_1SomeF16,
    /// All tensors are mostly stored as `Q4_2`, except for the 1D tensors (32-bit).
    MostlyQ4_2,
    /// All tensors are mostly stored as `Q8_0`, except for the 1D tensors (32-bit).
    MostlyQ8_0,
    /// All tensors are mostly stored as `Q5_0`, except for the 1D tensors (32-bit).
    MostlyQ5_0,
    /// All tensors are mostly stored as `Q5_1`, except for the 1D tensors (32-bit).
    MostlyQ5_1,
}

impl From<ElementType> for llm::FileTypeFormat {
    fn from(element_type: ElementType) -> Self {
        match element_type {
            ElementType::F32 => llm::FileTypeFormat::F32,
            ElementType::MostlyF16 => llm::FileTypeFormat::MostlyF16,
            ElementType::MostlyQ4_0 => llm::FileTypeFormat::MostlyQ4_0,
            ElementType::MostlyQ4_1 => llm::FileTypeFormat::MostlyQ4_1,
            ElementType::MostlyQ4_1SomeF16 => llm::FileTypeFormat::MostlyQ4_1SomeF16,
            ElementType::MostlyQ4_2 => llm::FileTypeFormat::MostlyQ4_2,
            ElementType::MostlyQ8_0 => llm::FileTypeFormat::MostlyQ8_0,
            ElementType::MostlyQ5_0 => llm::FileTypeFormat::MostlyQ5_0,
            ElementType::MostlyQ5_1 => llm::FileTypeFormat::MostlyQ5_1,
        }
    }
}

/// Not implemented yet.
#[napi(js_name = "convert")]
pub async fn convert(path: String, _element_type: ElementType) -> Result<()> {
    let handle = tokio::task::spawn_blocking(move || {
        let path = Path::new(path.as_str());
        println!("path: {:?}", path);
        // convert_pth_to_ggml is removed from llm
        // convert_pth_to_ggml(path, element_type.into());
    })
    .await;
    match handle {
        Ok(_) => Ok(()),
        Err(_) => Err(napi::Error::new(
            napi::Status::GenericFailure,
            "Failed to convert model".to_string(),
        )),
    }
}

#[napi]
pub struct LLM {
    llm: Arc<context::LLMContext>,
}

/// LLM class is a Rust wrapper for llm-rs.
#[napi]
impl LLM {
    /// Create a new LLM instance.
    #[napi]
    pub async fn load(config: ModelLoad, enable_logger: bool) -> Result<LLM> {
        let logger = LLamaLogger::get_singleton();
        logger.set_enabled(enable_logger);

        let llm = LLMContext::load_model(&config).await?;

        Ok(LLM { llm: Arc::new(llm) })
    }

    /// Get the tokenized result as number array, the result will be returned as Promise of number array.
    #[napi]
    pub async fn tokenize(&self, params: String) -> Result<Vec<i32>> {
        self.llm.tokenize(&params).await
    }

    /// Get the embedding result as number array, the result will be returned as Promise of number array.
    #[napi]
    pub async fn get_word_embeddings(
        &self,
        #[napi(ts_arg_type = "Partial<Generate>")] params: serde_json::Value,
    ) -> Result<Vec<f64>> {
        let params = serde_json::from_value::<Generate>(params).unwrap();
        self.llm.get_word_embedding(&params).await
    }

    /// Streaming the inference result as string, the result will be passed to the callback function. Will return a function to abort the inference.
    #[napi(ts_return_type = "() => void")]
    pub fn inference(
        &self,
        env: Env,
        #[napi(ts_arg_type = "Partial<Generate>")] params: serde_json::Value,
        #[napi(ts_arg_type = "(result: InferenceResult) => void")] callback: JsFunction,
    ) -> Result<JsFunction> {
        let params = serde_json::from_value::<Generate>(params).unwrap();
        let tsfn: ThreadsafeFunction<InferenceResult, ErrorStrategy::Fatal> = callback
            .create_threadsafe_function(0, |ctx: ThreadSafeCallContext<InferenceResult>| {
                Ok(vec![ctx.value])
            })?;

        let llm = self.llm.clone();

        let running = Arc::new(Mutex::new(true));
        {
            let running = running.clone();
            tokio::task::spawn_blocking(move || {
                llm.inference(&params, |result| {
                    let running = running.blocking_lock();
                    tsfn.call(result, ThreadsafeFunctionCallMode::NonBlocking);
                    if *running {
                        InferenceFeedback::Continue
                    } else {
                        InferenceFeedback::Halt
                    }
                })
                .map_err(|e| {
                    log::error!("Error in inference: {:?}", e);
                })
            });
        }

        env.create_function_from_closure("abort_inference", move |_| {
            let mut running = running.blocking_lock();
            *running = false;
            Ok(())
        })
    }
}
