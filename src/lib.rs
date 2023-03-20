#![deny(clippy::all)]

#[macro_use]
extern crate napi_derive;

mod llama;
use llama::{InferenceData, InferenceParams, LLamaChannel, LoadParams};

use napi::{
  bindgen_prelude::*,
  threadsafe_function::{
    ErrorStrategy, ThreadSafeCallContext, ThreadsafeFunction, ThreadsafeFunctionCallMode,
  },
  JsFunction,
};
use std::{
  sync::{
    mpsc::{channel, Receiver},
    Arc, Mutex,
  },
  thread,
};

#[napi]
pub struct LLama {
  rx: Arc<Mutex<Receiver<InferenceData>>>,
  llama_channel: LLamaChannel,
}

#[napi]
impl LLama {
  #[napi]
  pub fn new() -> Self {
    let (tx, rx) = channel::<InferenceData>();

    let llama_channel = LLamaChannel::new(tx.clone());

    LLama {
      rx: Arc::new(Mutex::new(rx)),
      llama_channel,
    }
  }

  #[napi]
  pub fn load_model(&mut self, params: LoadParams) -> Result<()> {
    self.llama_channel.load_model(params);
    Ok(())
  }

  #[napi]
  pub fn inference(&mut self, params: InferenceParams) -> Result<()> {
    self.llama_channel.inference(params);
    Ok(())
  }

  #[napi(ts_args_type = "callback: (err: null | Error, result: { token: string; completed: number }) => void")]
  pub fn on_generated(&self, callback: JsFunction) -> Result<()> {
    let tsfn: ThreadsafeFunction<InferenceData, ErrorStrategy::CalleeHandled> = callback
      .create_threadsafe_function(0, |ctx: ThreadSafeCallContext<InferenceData>| {
        // ctx.env.create_string_from_std(ctx.value).map(|v| vec![v])
        let mut obj = ctx.env.create_object().unwrap();
        let token = ctx.env.create_string(ctx.value.token.as_str()).unwrap();
        let completed = ctx
          .env
          .create_int32(if ctx.value.completed { 1 } else { 0 }).unwrap();
        obj.set_named_property("token", token).unwrap();
        obj.set_named_property("completed", completed).unwrap();
        Ok(vec![obj])
      })?;
    let rx = self.rx.clone();

    thread::spawn(move || loop {
      let rx = rx.lock().unwrap();
      match rx.try_recv() {
        Ok(str) => {
          tsfn.call(Ok(str), ThreadsafeFunctionCallMode::NonBlocking);
        }
        Err(_) => {
          std::thread::yield_now();
        }
      }
    });

    Ok(())
  }
}
