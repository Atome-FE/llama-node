#![deny(clippy::all)]
#![allow(clippy::enum_variant_names)]

#[macro_use]
extern crate napi_derive;

mod llama;
mod types;

use std::{
  path::Path,
  sync::{mpsc::channel, Arc},
  thread, time,
};

use llama::LLamaChannel;
use llama_rs::convert::convert_pth_to_ggml;
use types::{
  EmbeddingResult, InferenceResult, InferenceResultType, LLamaConfig, LLamaInferenceArguments,
  LoadModelResult, TokenizeResult,
};

use napi::{
  bindgen_prelude::*,
  threadsafe_function::{
    ErrorStrategy, ThreadSafeCallContext, ThreadsafeFunction, ThreadsafeFunctionCallMode,
  },
  JsFunction,
};

#[napi]
pub enum ElementType {
  /// Quantized 4-bit (type 0).
  Q4_0,
  /// Quantized 4-bit (type 1); used by GPTQ.
  Q4_1,
  /// Float 16-bit.
  F16,
  /// Float 32-bit.
  F32,
}

impl From<ElementType> for llama_rs::ElementType {
  fn from(element_type: ElementType) -> Self {
    match element_type {
      ElementType::Q4_0 => llama_rs::ElementType::Q4_0,
      ElementType::Q4_1 => llama_rs::ElementType::Q4_1,
      ElementType::F16 => llama_rs::ElementType::F16,
      ElementType::F32 => llama_rs::ElementType::F32,
    }
  }
}

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
#[derive(Clone)]
pub struct LLama {
  llama_channel: Arc<LLamaChannel>,
}

#[napi]
impl LLama {
  #[napi]
  pub fn enable_logger() {
    env_logger::builder()
      .filter_level(log::LevelFilter::Info)
      .parse_default_env()
      .init();
  }

  #[napi]
  pub fn create(config: LLamaConfig) -> Result<LLama> {
    let (load_result_sender, load_result_receiver) = channel::<LoadModelResult>();

    let llama_channel = LLamaChannel::new();

    llama_channel.load_model(config, load_result_sender);

    // currently this loop blocked main thread, will try improve in the future
    'waiting_load: loop {
      let recv = load_result_receiver.recv();
      match recv {
        Ok(r) => {
          if r.error {
            return Err(Error::new(
              Status::InvalidArg,
              r.message.unwrap_or("Unknown Error".to_string()),
            ));
          }
          break 'waiting_load;
        }
        _ => {
          thread::yield_now();
        }
      }
    }

    Ok(LLama { llama_channel })
  }

  #[napi]
  pub fn tokenize(
    &self,
    params: String,
    #[napi(ts_arg_type = "(result: TokenizeResult) => void")] callback: JsFunction,
  ) -> Result<()> {
    let (tokenize_sender, tokenize_receiver) = channel::<TokenizeResult>();

    let tsfn: ThreadsafeFunction<TokenizeResult, ErrorStrategy::Fatal> = callback
      .create_threadsafe_function(0, |ctx: ThreadSafeCallContext<TokenizeResult>| {
        Ok(vec![ctx.value])
      })?;

    let llama_channel = self.llama_channel.clone();

    llama_channel.tokenize(&params, tokenize_sender);

    thread::spawn(move || {
      'waiting_tokenize: loop {
        let recv = tokenize_receiver.recv();
        match recv {
          Ok(callback) => {
            tsfn.call(callback, ThreadsafeFunctionCallMode::Blocking);
            break 'waiting_tokenize;
          }
          _ => {
            thread::yield_now();
          }
        }
      }
      thread::sleep(time::Duration::from_millis(300)); // wait for end signal
      tsfn.abort().unwrap();
    });

    Ok(())
  }

  #[napi]
  pub fn get_word_embeddings(
    &self,
    params: LLamaInferenceArguments,
    #[napi(ts_arg_type = "(result: EmbeddingResult) => void")] callback: JsFunction,
  ) -> Result<()> {
    let (embedding_sender, embedding_receiver) = channel::<EmbeddingResult>();

    let tsfn: ThreadsafeFunction<EmbeddingResult, ErrorStrategy::Fatal> = callback
      .create_threadsafe_function(0, |ctx: ThreadSafeCallContext<EmbeddingResult>| {
        Ok(vec![ctx.value])
      })?;

    let llama_channel = self.llama_channel.clone();

    llama_channel.get_word_embedding(params, embedding_sender);

    thread::spawn(move || {
      'waiting_embedding: loop {
        let recv = embedding_receiver.recv();
        match recv {
          Ok(callback) => {
            tsfn.call(callback, ThreadsafeFunctionCallMode::Blocking);
            break 'waiting_embedding;
          }
          _ => {
            thread::yield_now();
          }
        }
      }
      thread::sleep(time::Duration::from_millis(300)); // wait for end signal
      tsfn.abort().unwrap();
    });

    Ok(())
  }

  #[napi]
  pub fn inference(
    &self,
    params: LLamaInferenceArguments,
    #[napi(ts_arg_type = "(result: InferenceResult) => void")] callback: JsFunction,
  ) -> Result<()> {
    let (inference_sender, inference_receiver) = channel::<InferenceResult>();

    let tsfn: ThreadsafeFunction<InferenceResult, ErrorStrategy::Fatal> = callback
      .create_threadsafe_function(0, |ctx: ThreadSafeCallContext<InferenceResult>| {
        Ok(vec![ctx.value])
      })?;

    let llama_channel = self.llama_channel.clone();

    llama_channel.inference(params, inference_sender);

    thread::spawn(move || {
      'waiting_inference: loop {
        let recv = inference_receiver.recv();
        match recv {
          Ok(callback) => match callback.r#type {
            InferenceResultType::End => {
              tsfn.call(callback, ThreadsafeFunctionCallMode::Blocking);
              break 'waiting_inference;
            }
            _ => {
              tsfn.call(callback, ThreadsafeFunctionCallMode::NonBlocking);
            }
          },
          _ => {
            thread::yield_now();
          }
        }
      }
      thread::sleep(time::Duration::from_millis(300)); // wait for end signal
      tsfn.abort().unwrap();
    });

    Ok(())
  }
}
