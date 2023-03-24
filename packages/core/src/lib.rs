#![deny(clippy::all)]

#[macro_use]
extern crate napi_derive;

mod llama;
mod types;

use std::{
  sync::{mpsc::channel, Arc},
  thread, time,
};

use llama::LLamaChannel;
use types::{InferenceResult, LLamaArguments, LLamaConfig, LoadModelResult};

use napi::{
  bindgen_prelude::*,
  threadsafe_function::{
    ErrorStrategy, ThreadSafeCallContext, ThreadsafeFunction, ThreadsafeFunctionCallMode,
  },
  JsFunction,
};

#[napi]
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

  #[napi(ts_args_type = "params: LLamaArguments,
    callback: (result: 
      { type: 'ERROR', message: string } |
      { type: 'DATA', data: InferenceToken } |
      { type: 'END' }
    ) => void")]
  pub fn inference(&self, params: LLamaArguments, callback: JsFunction) -> Result<()> {
    let (inference_sender, inference_receiver) = channel::<InferenceResult>();

    let tsfn: ThreadsafeFunction<InferenceResult, ErrorStrategy::Fatal> = callback
      .create_threadsafe_function(0, |ctx: ThreadSafeCallContext<InferenceResult>| {
        let mut obj = ctx.env.create_object().unwrap();
        match ctx.value.clone() {
          InferenceResult::InferenceData(it) => {
            let mut data = ctx.env.create_object().unwrap();
            let token = ctx.env.create_string(it.token.as_str()).unwrap();
            let completed = ctx
              .env
              .create_int32(if it.completed { 1 } else { 0 })
              .unwrap();
            data.set_named_property("token", token).unwrap();
            data.set_named_property("completed", completed).unwrap();
            obj
              .set_named_property("type", ctx.env.create_string("DATA").unwrap())
              .unwrap();
            obj.set_named_property("data", data).unwrap();
          }
          InferenceResult::InferenceError(err) => {
            let error = ctx.env.create_string(err.as_str()).unwrap();
            obj
              .set_named_property("type", ctx.env.create_string("ERROR").unwrap())
              .unwrap();
            obj.set_named_property("message", error).unwrap();
          }
          InferenceResult::InferenceEnd => {
            obj
              .set_named_property("type", ctx.env.create_string("END").unwrap())
              .unwrap();
          }
        };
        Ok(vec![obj])
      })?;

    let llama_channel = self.llama_channel.clone();

    let tsfn = tsfn.clone();

    llama_channel.inference(params, inference_sender);

    thread::spawn(move || {
      'waiting_inference: loop {
        let recv = inference_receiver.recv();
        match recv {
          Ok(callback) => match callback {
            InferenceResult::InferenceEnd => {
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
