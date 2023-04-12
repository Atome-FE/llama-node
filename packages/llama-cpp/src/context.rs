use std::{ffi::CStr, ptr::null_mut, slice};

use anyhow::Result;
use llm_chain_llama_sys::{
    llama_context, llama_context_default_params, llama_context_params, llama_eval, llama_free,
    llama_get_embeddings, llama_init_from_file, llama_n_embd, llama_print_system_info,
    llama_sample_top_p_top_k, llama_token, llama_token_to_str,
};

#[napi(object)]
#[derive(Debug, Clone)]
pub struct LlamaInvocation {
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
pub struct LlamaContextParams {
    pub n_ctx: i32,
    pub n_parts: i32,
    pub seed: i32,
    pub f16_kv: bool,
    pub logits_all: bool,
    pub vocab_only: bool,
    pub use_mlock: bool,
    pub embedding: bool,
    // pub use_mmap: bool,
}

impl LlamaContextParams {
    // Returns the default parameters or the user-specified parameters.
    pub(crate) fn or_default(params: &Option<LlamaContextParams>) -> llama_context_params {
        match params {
            Some(params) => params.clone().into(),
            None => unsafe { llama_context_default_params() },
        }
    }
}

impl From<LlamaContextParams> for llama_context_params {
    fn from(params: LlamaContextParams) -> Self {
        llama_context_params {
            n_ctx: params.n_ctx,
            n_parts: params.n_parts,
            seed: params.seed,
            f16_kv: params.f16_kv,
            logits_all: params.logits_all,
            vocab_only: params.vocab_only,
            use_mlock: params.use_mlock,
            embedding: params.embedding,
            progress_callback: None,
            progress_callback_user_data: null_mut(),
            // use_mmap: params.use_mmap,
        }
    }
}

// Represents the LLamaContext which wraps FFI calls to the llama.cpp library.
pub struct LLamaContext {
    ctx: *mut llama_context,
}

impl LLamaContext {
    // Creates a new LLamaContext from the specified file and configuration parameters.
    pub fn from_file_and_params(path: &str, params: &Option<LlamaContextParams>) -> Self {
        let params = LlamaContextParams::or_default(params);
        let ctx = unsafe { llama_init_from_file(path.as_ptr() as *const i8, params) };
        Self { ctx }
    }

    pub fn llama_print_system_info(&self) {
        let sys_info_c_str = unsafe { llama_print_system_info() };
        let sys_info = unsafe { CStr::from_ptr(sys_info_c_str) }
            .to_str()
            .unwrap()
            .to_owned();
        log::info!("{}", sys_info);
    }

    // Executes the LLama sampling process with the specified configuration.
    pub fn llama_sample(
        &self,
        last_n_tokens_data: &[llama_token],
        last_n_tokens_size: i32,
        input: &LlamaInvocation,
    ) -> i32 {
        let top_p = input.top_p as f32;
        let temp = input.temp as f32;
        let repeat_penalty = input.repeat_penalty as f32;
        unsafe {
            llama_sample_top_p_top_k(
                self.ctx,
                last_n_tokens_data.as_ptr(),
                last_n_tokens_size,
                input.top_k,
                top_p,
                temp,
                repeat_penalty,
            )
        }
    }

    pub fn llama_token_to_str(&self, token: &i32) -> String {
        let c_ptr = unsafe { llama_token_to_str(self.ctx, *token) };
        let native_string = unsafe { CStr::from_ptr(c_ptr) }
            .to_str()
            .unwrap()
            .to_owned();
        native_string
    }

    pub fn llama_get_embeddings(&self) -> Result<Vec<f32>, ()> {
        unsafe {
            let embd_size = llama_n_embd(self.ctx);
            let embd_ptr = llama_get_embeddings(self.ctx);
            if embd_ptr.is_null() {
                return Err(());
            }
            Ok(slice::from_raw_parts(embd_ptr, embd_size as usize).to_vec())
        }
    }

    // Evaluates the given tokens with the specified configuration.
    pub fn llama_eval(
        &self,
        tokens: &[llama_token],
        n_tokens: i32,
        n_past: i32,
        input: &LlamaInvocation,
    ) -> Result<(), ()> {
        let res =
            unsafe { llama_eval(self.ctx, tokens.as_ptr(), n_tokens, n_past, input.n_threads) };
        if res == 0 {
            Ok(())
        } else {
            Err(())
        }
    }
}

// Provides thread-safe behavior for LLamaContext.
unsafe impl Send for LLamaContext {}
unsafe impl Sync for LLamaContext {}

// Enables dereferencing LLamaContext to access the underlying *mut llama_context.
impl std::ops::Deref for LLamaContext {
    type Target = *mut llama_context;
    fn deref(&self) -> &*mut llama_context {
        &self.ctx
    }
}

// Handles proper cleanup of the llama_context when the LLamaContext is dropped.
impl Drop for LLamaContext {
    fn drop(&mut self) {
        unsafe { llama_free(self.ctx) };
    }
}
