#![deny(clippy::all)]
#![allow(clippy::enum_variant_names)]

#[macro_use]
extern crate napi_derive;

mod llama;
mod tokenizer;
mod output;

use std::io::Write;

use llama::{LLamaContext, LlamaContextParams};
use napi::{
    bindgen_prelude::*,
    threadsafe_function::{
        ErrorStrategy, ThreadSafeCallContext, ThreadsafeFunction, ThreadsafeFunctionCallMode,
    },
    JsFunction,
};
use tokenizer::{embedding_to_output, llama_token_eos, tokenize};

// #[napi(object)]
// pub struct LlamaInvocation {
//     pub n_threads: i32,
//     pub n_tok_predict: u32,
//     pub top_k: i32,
//     pub top_p: f64,
//     pub temp: f64,
//     pub repeat_penalty: f64,
//     pub stop_sequence: String,
//     pub prompt: String,
// }

#[napi]
pub struct LLama {
    context: LLamaContext,
    context_params: Option<LlamaContextParams>,
}

#[napi]
impl LLama {
    #[napi]
    pub fn new(path: String, params: Option<LlamaContextParams>) -> Self {
        Self {
            context: LLamaContext::from_file_and_params(&path, &params),
            context_params: params,
        }
    }

    #[napi]
    pub fn run(
        &self,
        input: llama::LlamaInvocation,
        // callback: &Option<fn(&Output)>,
        callback: JsFunction
    ) {

        // let tsfn = callback.create_threadsafe_function(0, callback)

        let input_ctx = &self.context;
        let context_params = &self.context_params;

        let context_params_c = LlamaContextParams::or_default(context_params);
        // Tokenize the stop sequence and input prompt.
        let tokenized_stop_prompt = tokenize(
            input_ctx,
            input.stop_sequence.as_str(),
            context_params_c.n_ctx as usize,
            false,
        )
        .unwrap();
        let tokenized_input = tokenize(
            input_ctx,
            input.prompt.as_str(),
            context_params_c.n_ctx as usize,
            true,
        )
        .unwrap();

        // Embd contains the prompt and the completion. The longer the prompt, the shorter the completion.
        let mut embd = tokenized_input.clone();
        embd.resize(context_params_c.n_ctx as usize, 0);

        // Evaluate the prompt in full.
        input_ctx
            .llama_eval(
                tokenized_input.as_slice(),
                tokenized_input.len() as i32,
                0,
                &input,
            )
            .unwrap();
        let token_eos = llama_token_eos();

        // Generate remaining tokens.
        let mut n_remaining = context_params_c.n_ctx - tokenized_input.len() as i32;
        let mut n_used = tokenized_input.len() - 1;
        let mut stop_sequence_i = 0;
        while n_remaining > 0 {
            let tok = input_ctx.llama_sample(embd.as_slice(), n_used as i32, &input);
            n_used += 1;
            n_remaining -= 1;
            embd[n_used] = tok;
            if tok == token_eos {
                break;
            }
            if input.n_tok_predict != 0 && n_used > input.n_tok_predict as usize + tokenized_input.len() - 1
            {
                break;
            }
            if tok == tokenized_stop_prompt[stop_sequence_i] {
                stop_sequence_i += 1;
                if stop_sequence_i >= tokenized_stop_prompt.len() {
                    break;
                }
            } else {
                stop_sequence_i = 0;
            }
            input_ctx
                .llama_eval(&embd[n_used..], 1, n_used as i32, &input)
                .unwrap();

            
            // if let Some(callback) = callback {
                let output = input_ctx.llama_token_to_str(&embd[n_used]);
                print!("{}", output);
                std::io::stdout().flush().unwrap();
            //     callback(&output.into());
            // }
        }
    }
}
