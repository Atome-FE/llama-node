// #![deny(clippy::all)]

#[macro_use]
extern crate napi_derive;

mod context;
mod llama;
mod output;
mod tokenizer;

use std::sync::{mpsc::channel, Arc};

use context::{LlamaContextParams, LlamaInvocation};
use llama::LLamaChannel;
use napi::{
    bindgen_prelude::*,
    threadsafe_function::{ErrorStrategy, ThreadsafeFunction, ThreadsafeFunctionCallMode},
    JsFunction,
};

#[napi]
pub struct LLama {
    llama_channel: Arc<LLamaChannel>,
}

#[napi]
impl LLama {
    #[napi]
    pub fn load(
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

        let (load_result_sender, load_result_receiver) = channel::<bool>();
        let llama_channel = LLamaChannel::new(path, params, load_result_sender, enable_logger);
        'waiting_load: loop {
            let recv = load_result_receiver.recv();
            match recv {
                Ok(r) => {
                    if !r {
                        return Err(Error::new(Status::InvalidArg, "Load error".to_string()));
                    }
                    break 'waiting_load;
                }
                _ => {
                    std::thread::yield_now();
                }
            }
        }
        Ok(Self { llama_channel })
    }

    #[napi]
    pub fn inference(&self, input: LlamaInvocation, callback: JsFunction) -> Result<()> {
        let tsfn: ThreadsafeFunction<String, ErrorStrategy::Fatal> = callback
            .create_threadsafe_function(0, |ctx| {
                let value = ctx.value;
                let data = ctx.env.create_string_from_std(value).unwrap();
                Ok(vec![data])
            })?;
        let (inference_sender, inference_receiver) = channel();
        let llama_channel = self.llama_channel.clone();

        llama_channel.inference(input, inference_sender);

        std::thread::spawn(move || loop {
            let result = inference_receiver.recv();
            match result {
                Ok(result) => {
                    tsfn.call(result, ThreadsafeFunctionCallMode::NonBlocking);
                }
                Err(_) => {
                    break;
                }
            }
        });

        Ok(())
    }
}
