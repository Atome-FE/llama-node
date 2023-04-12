#![deny(clippy::all)]

#[macro_use]
extern crate napi_derive;

mod context;
mod llama;
mod output;
mod tokenizer;
mod types;

use std::{
    sync::{mpsc::channel, Arc},
    thread, time,
};

use context::{LlamaContextParams, LlamaInvocation};
use llama::LLamaChannel;
use napi::{
    bindgen_prelude::*,
    threadsafe_function::{
        ErrorStrategy, ThreadSafeCallContext, ThreadsafeFunction, ThreadsafeFunctionCallMode,
    },
    JsFunction,
};
use types::{InferenceResult, TokenizeResult, EmbeddingResult};

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
                    thread::yield_now();
                }
            }
        }
        Ok(Self { llama_channel })
    }

    #[napi(ts_args_type = "input: LlamaInvocation,
        callback: (result: EmbeddingResult) => void")]
    pub fn get_word_embedding(&self, input: LlamaInvocation, callback: JsFunction) -> Result<()> {
        let tsfn: ThreadsafeFunction<EmbeddingResult, ErrorStrategy::Fatal> =
            callback.create_threadsafe_function(0, |ctx| Ok(vec![ctx.value]))?;
        let (embeddings_sender, embeddings_receiver) = channel();
        let llama_channel = self.llama_channel.clone();

        llama_channel.embedding(input, embeddings_sender);

        thread::spawn(move || {
            loop {
                let result = embeddings_receiver.recv();
                match result {
                    Ok(result) => {
                        tsfn.call(result, ThreadsafeFunctionCallMode::NonBlocking);
                    }
                    Err(_) => {
                        break;
                    }
                }
            }
            thread::sleep(time::Duration::from_millis(300)); // wait for end signal
            tsfn.abort().unwrap();
        });

        Ok(())
    }

    #[napi(ts_args_type = "params: string,
    nCtx: number,
    callback: (result:
      { type: TokenizeResultType, data: number[] }
    ) => void")]
    pub fn tokenize(&self, params: String, n_ctx: i32, callback: JsFunction) -> Result<()> {
        let (tokenize_sender, tokenize_receiver) = channel::<TokenizeResult>();

        let tsfn: ThreadsafeFunction<TokenizeResult, ErrorStrategy::Fatal> = callback
            .create_threadsafe_function(0, |ctx: ThreadSafeCallContext<TokenizeResult>| {
                Ok(vec![ctx.value])
            })?;

        let llama_channel = self.llama_channel.clone();

        llama_channel.tokenize(params, n_ctx as usize, tokenize_sender);

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

    #[napi(ts_args_type = "input: LlamaInvocation,
        callback: (result: InferenceResult) => void")]
    pub fn inference(&self, input: LlamaInvocation, callback: JsFunction) -> Result<()> {
        let tsfn: ThreadsafeFunction<InferenceResult, ErrorStrategy::Fatal> =
            callback.create_threadsafe_function(0, |ctx| Ok(vec![ctx.value]))?;
        let (inference_sender, inference_receiver) = channel();
        let llama_channel = self.llama_channel.clone();

        llama_channel.inference(input, inference_sender);

        thread::spawn(move || {
            loop {
                let result = inference_receiver.recv();
                match result {
                    Ok(result) => {
                        tsfn.call(result, ThreadsafeFunctionCallMode::NonBlocking);
                    }
                    Err(_) => {
                        break;
                    }
                }
            }
            thread::sleep(time::Duration::from_millis(300)); // wait for end signal
            tsfn.abort().unwrap();
        });

        Ok(())
    }
}
