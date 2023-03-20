#![deny(clippy::all)]

#[macro_use]
extern crate napi_derive;

mod llama;
use llama::{InferenceCallback, LLamaArguments, LLamaChannel, LLamaConfig};

use napi::{
  bindgen_prelude::*,
  threadsafe_function::{
    ErrorStrategy, ThreadSafeCallContext, ThreadsafeFunction, ThreadsafeFunctionCallMode,
  },
  JsFunction,
};
use std::{
  sync::{
    mpsc::{channel, Receiver, Sender, TryRecvError},
    Arc, Mutex,
  },
  thread,
};

#[napi]
pub struct LLama {
  inference_receiver: Arc<Mutex<Receiver<InferenceCallback>>>,
  inference_sender: Sender<InferenceCallback>,
  llama_channel: LLamaChannel,
}

#[napi]
impl LLama {
  #[napi]
  pub fn create(config: LLamaConfig) -> Self {
    let (tx, rx) = channel::<InferenceCallback>();

    let llama_channel = LLamaChannel::new(tx.clone());

    llama_channel.load_model(config);

    LLama {
      inference_receiver: Arc::new(Mutex::new(rx)),
      llama_channel,
      inference_sender: tx.clone(),
    }
  }

  #[napi]
  pub fn inference(&mut self, params: LLamaArguments) -> Result<()> {
    self.llama_channel.inference(params);
    Ok(())
  }

  #[napi]
  pub fn terminate(&self) {
    self
      .inference_sender
      .send(InferenceCallback::Terminate)
      .unwrap();
    self.llama_channel.terminate();
  }

  #[napi(
    ts_args_type = "callback: (result: { type: 'ERROR', message: string } | { type: 'DATA', data: { token: string; completed: number } }) => void"
  )]
  pub fn on_generated(&self, callback: JsFunction) -> Result<()> {
    let tsfn: ThreadsafeFunction<InferenceCallback, ErrorStrategy::Fatal> = callback
      .create_threadsafe_function(0, |ctx: ThreadSafeCallContext<InferenceCallback>| {
        let mut obj = ctx.env.create_object().unwrap();

        return match ctx.value {
          InferenceCallback::InferenceData(it) => {
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
            Ok(vec![obj])
          }
          InferenceCallback::InferenceError(error) => {
            let error = ctx.env.create_string(error.as_str()).unwrap();
            obj
              .set_named_property("type", ctx.env.create_string("ERROR").unwrap())
              .unwrap();
            obj.set_named_property("message", error).unwrap();
            Ok(vec![obj])
          }
          _ => {
            unreachable!("Termination is not reachable")
          }
        };
      })?;
    let inference_receiver = self.inference_receiver.clone();

    thread::spawn(move || loop {
      let inference_receiver = inference_receiver.lock().unwrap();
      match inference_receiver.try_recv() {
        Err(TryRecvError::Disconnected) | Ok(InferenceCallback::Terminate) => {
          tsfn.abort().unwrap();
          break;
        }
        Ok(callback) => {
          tsfn.call(callback, ThreadsafeFunctionCallMode::NonBlocking);
        }
        Err(_) => {
          std::thread::yield_now();
        }
      }
    });

    Ok(())
  }
}
