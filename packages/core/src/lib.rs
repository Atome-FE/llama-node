#![deny(clippy::all)]
#![allow(clippy::enum_variant_names)]

#[macro_use]
extern crate napi_derive;

mod llama;
mod types;

use std::{path::Path, sync::Arc};

use llama::LLamaInternal;
use llama_rs::convert::convert_pth_to_ggml;
use types::{EmbeddingResult, InferenceResult, LLamaConfig, LLamaInferenceArguments};

use napi::{
  bindgen_prelude::*,
  threadsafe_function::{
    ErrorStrategy, ThreadSafeCallContext, ThreadsafeFunction, ThreadsafeFunctionCallMode,
  },
  JsFunction,
};

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
  /// All tensors are mostly stored as `Q4_3`, except for the 1D tensors (32-bit).
  MostlyQ4_3,
}

impl From<ElementType> for llama_rs::FileType {
  fn from(element_type: ElementType) -> Self {
    match element_type {
      ElementType::F32 => llama_rs::FileType::F32,
      ElementType::MostlyF16 => llama_rs::FileType::MostlyF16,
      ElementType::MostlyQ4_0 => llama_rs::FileType::MostlyQ4_0,
      ElementType::MostlyQ4_1 => llama_rs::FileType::MostlyQ4_1,
      ElementType::MostlyQ4_1SomeF16 => llama_rs::FileType::MostlyQ4_1SomeF16,
      ElementType::MostlyQ4_2 => llama_rs::FileType::MostlyQ4_2,
      ElementType::MostlyQ4_3 => llama_rs::FileType::MostlyQ4_3,
    }
  }
}

/// Not implemented yet.
#[napi(js_name = "convert")]
pub async fn convert(path: String, element_type: ElementType) -> Result<()> {
  let handle = tokio::task::spawn_blocking(move || {
    let path = Path::new(path.as_str());
    println!("path: {:?}", path);
    convert_pth_to_ggml(path, element_type.into());
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

#[napi(js_name = "LLama")]
pub struct LLama {
  async_llama: Arc<llama::LLamaInternal>,
}

/// LLama class is a Rust wrapper for llama-rs.
#[napi]
impl LLama {
  /// Enable logger.
  #[napi]
  pub fn enable_logger() {
    env_logger::builder()
      .filter_level(log::LevelFilter::Info)
      .parse_default_env()
      .init();
  }

  /// Create a new LLama instance.
  #[napi]
  pub async fn create(config: LLamaConfig) -> Result<LLama> {
    let mut async_llama = LLamaInternal { model: None };

    async_llama.load_model(&config).await?;

    Ok(LLama {
      async_llama: Arc::new(async_llama),
    })
  }

  /// Get the tokenized result as number array, the result will be passed to the callback function.
  #[napi]
  pub async fn tokenize(&self, params: String) -> Result<Vec<i32>> {
    self.async_llama.tokenize(&params).await
  }

  /// Get the embedding result as number array, the result will be passed to the callback function.
  #[napi]
  pub async fn get_word_embeddings(
    &self,
    params: LLamaInferenceArguments,
  ) -> Result<EmbeddingResult> {
    self.async_llama.get_word_embedding(&params).await
  }

  /// Streaming the inference result as string, the result will be passed to the callback function.
  #[napi]
  pub fn inference(
    &self,
    params: LLamaInferenceArguments,
    #[napi(ts_arg_type = "(result: InferenceResult) => void")] callback: JsFunction,
  ) -> Result<()> {
    let tsfn: ThreadsafeFunction<InferenceResult, ErrorStrategy::Fatal> = callback
      .create_threadsafe_function(0, |ctx: ThreadSafeCallContext<InferenceResult>| {
        Ok(vec![ctx.value])
      })?;

    let async_llama = self.async_llama.clone();

    tokio::spawn(async move {
      async_llama
        .inference(&params, |r| {
          tsfn.call(r, ThreadsafeFunctionCallMode::NonBlocking);
        })
        .await;
    });

    Ok(())
  }
}
