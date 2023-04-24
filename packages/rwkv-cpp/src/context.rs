use std::convert::TryInto;
use std::{ffi::CStr, ptr::null_mut, slice};
use tokenizers::tokenizer::Tokenizer;

use anyhow::Result;
use rwkv_sys::{
    // llama_context, llama_context_default_params, llama_context_params, llama_eval, llama_free,
    // llama_get_embeddings, llama_init_from_file, llama_n_embd, llama_print_system_info,
    // llama_sample_top_p_top_k, llama_token, llama_token_to_str,
    rwkv_context,
    rwkv_eval,
    rwkv_free,
    rwkv_get_logits_buffer_element_count,
    rwkv_get_state_buffer_element_count,
    rwkv_get_system_info_string,
    rwkv_init_from_file,
};

#[napi(object)]
#[derive(Debug, Clone)]
pub struct RWKVInvocation {
    pub n_threads: i32,
    pub n_tok_predict: i32,
    pub top_k: i32,
    pub top_p: f64,
    pub temp: f64,
    pub repeat_penalty: f64,
    pub stop_sequence: Option<String>,
    pub prompt: String,
}

// Represents the configuration parameters for a LLamaContext.
#[napi(object)]
#[derive(Debug, Clone)]
pub struct RWKVContextParams {
    pub n_ctx: i32,
    pub n_parts: i32,
    pub seed: i32,
    pub f16_kv: bool,
    pub logits_all: bool,
    pub vocab_only: bool,
    pub use_mlock: bool,
    pub embedding: bool,
    pub use_mmap: bool,
}

// impl LlamaContextParams {
//     // Returns the default parameters or the user-specified parameters.
//     pub(crate) fn or_default(params: &Option<LlamaContextParams>) -> llama_context_params {
//         match params {
//             Some(params) => params.clone().into(),
//             None => unsafe { llama_context_default_params() },
//         }
//     }
// }

// impl From<LlamaContextParams> for llama_context_params {
//     fn from(params: LlamaContextParams) -> Self {
//         llama_context_params {
//             n_ctx: params.n_ctx,
//             n_parts: params.n_parts,
//             seed: params.seed,
//             f16_kv: params.f16_kv,
//             logits_all: params.logits_all,
//             vocab_only: params.vocab_only,
//             use_mlock: params.use_mlock,
//             embedding: params.embedding,
//             progress_callback: None,
//             progress_callback_user_data: null_mut(),
//             use_mmap: params.use_mmap,
//         }
//     }
// }

// Represents the LLamaContext which wraps FFI calls to the llama.cpp library.
pub struct RWKVContext {
    ctx: *mut rwkv_context,
    pub tokenizer: Tokenizer,
    state_buffer_element_count: u32,
    logits_buffer_element_count: u32,
    model_tokens: Vec<u32>,
    model_state: Vec<f32>,
    pub logits: Vec<f32>,
}

impl RWKVContext {
    // Creates a new LLamaContext from the specified file and configuration parameters.
    pub fn from_file_and_params(model_path: &str, tokenizer_path: &str, params: u32) -> Self {
        // let params = LlamaContextParams::or_default(params);
        let ctx = unsafe { rwkv_init_from_file(model_path.as_ptr() as *const i8, params) };
        let state_buffer_element_count = unsafe { rwkv_get_state_buffer_element_count(ctx) };
        let logits_buffer_element_count = unsafe { rwkv_get_logits_buffer_element_count(ctx) };
        let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();
        Self {
            ctx,
            tokenizer,
            state_buffer_element_count,
            logits_buffer_element_count,
            model_tokens: vec![],
            model_state: vec![],
            logits: vec![],
        }
    }

    pub fn rwkv_print_system_info_string(&self) {
        let sys_info_c_str = unsafe { rwkv_get_system_info_string() };
        let sys_info = unsafe { CStr::from_ptr(sys_info_c_str) }
            .to_str()
            .unwrap()
            .to_owned();
        log::info!("{}", sys_info);
    }

    // // Executes the LLama sampling process with the specified configuration.
    // pub fn llama_sample(
    //     &self,
    //     last_n_tokens_data: &[llama_token],
    //     last_n_tokens_size: i32,
    //     input: &LlamaInvocation,
    // ) -> i32 {
    //     let top_p = input.top_p as f32;
    //     let temp = input.temp as f32;
    //     let repeat_penalty = input.repeat_penalty as f32;
    //     unsafe {
    //         llama_sample_top_p_top_k(
    //             self.ctx,
    //             last_n_tokens_data.as_ptr(),
    //             last_n_tokens_size,
    //             input.top_k,
    //             top_p,
    //             temp,
    //             repeat_penalty,
    //         )
    //     }
    // }

    pub fn rwkv_token_to_str(&self, token: &i32) -> Option<String> {
        let tokenizer = &self.tokenizer;
        tokenizer
            .decode(vec![(*token).try_into().unwrap()], false)
            .map(Some)
            .unwrap_or(None)
    }

    pub fn process_tokens(&mut self, tokens: &[u32]) {
        self.model_tokens.append(&mut tokens.to_vec());
        println!("model_tokens: {:?}", self.model_tokens);
        for token in tokens.iter() {
            let this = self as *mut Self;
            let model_state = &mut self.model_state as *mut Vec<f32>;
            let logits = &mut self.logits as *mut Vec<f32>;
            let (new_model_state, new_logits) = unsafe {
                (*this)
                    .rwkv_eval(
                        *token as i32,
                        model_state.as_mut().unwrap(),
                        model_state.as_mut().unwrap(),
                        logits.as_mut().unwrap(),
                    )
                    .unwrap()
            };
            // println!("new_model_state: {:?}", new_model_state);
            println!("new_logits: {:?}", new_logits); // TODO: debug this, no output
            self.model_state = new_model_state.to_vec();
            self.logits = new_logits.to_vec();
        }
    }

    // Evaluates the given tokens with the specified configuration.
    pub fn rwkv_eval<'a>(
        &'a mut self,
        token: i32,
        state_in: &'a mut [f32],
        state_out: &'a mut [f32],
        logits_out: &'a mut [f32],
    ) -> Result<(&'a mut [f32], &'a mut [f32]), ()> {
        let res = unsafe {
            rwkv_eval(
                self.ctx,
                token,
                state_in.as_mut_ptr(),
                state_out.as_mut_ptr(),
                logits_out.as_mut_ptr(),
            )
        };
        if res {
            Ok((logits_out, state_out))
        } else {
            Err(())
        }
    }
}

// Provides thread-safe behavior for RWKVContext.
unsafe impl Send for RWKVContext {}
unsafe impl Sync for RWKVContext {}

// Enables dereferencing RWKVContext to access the underlying *mut llama_context.
impl std::ops::Deref for RWKVContext {
    type Target = *mut rwkv_context;
    fn deref(&self) -> &*mut rwkv_context {
        &self.ctx
    }
}

// Handles proper cleanup of the rwkv_context when the RWKVContext is dropped.
impl Drop for RWKVContext {
    fn drop(&mut self) {
        unsafe { rwkv_free(self.ctx) };
    }
}
