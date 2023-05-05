use std::convert::TryInto;
use std::ffi::CStr;
use std::ptr::null_mut;
use tokenizers::tokenizer::Tokenizer;

use anyhow::Result;
use rwkv_sys::{
    rwkv_context, rwkv_eval, rwkv_free, rwkv_get_logits_buffer_element_count,
    rwkv_get_state_buffer_element_count, rwkv_get_system_info_string, rwkv_init_from_file,
};

#[napi(object)]
#[derive(Debug, Clone)]
pub struct RWKVInvocation {
    pub max_predict_length: i32,
    pub top_p: f64,
    pub temp: f64,
    pub end_token: Option<i32>,
    pub seed: Option<i32>,
    pub prompt: String,
}

// Represents the RWKVContext which wraps FFI calls to the rwkv.cpp library.
#[derive(Clone)]
pub struct RWKVContext {
    ctx: *mut rwkv_context,
    pub tokenizer: Tokenizer,
    state_buffer_element_count: u32,
    logits_buffer_element_count: u32,
    model_tokens: Vec<u32>,
    model_state: Vec<f32>,
    pub logits: Vec<f32>,
    is_first: bool,
}

impl RWKVContext {
    // Creates a new RWKVContext from the specified file and configuration parameters.
    pub fn from_file_and_params(model_path: &str, tokenizer_path: &str, n_threads: u32) -> Self {
        let ctx = unsafe { rwkv_init_from_file(model_path.as_ptr() as *const i8, n_threads) };
        let state_buffer_element_count = unsafe { rwkv_get_state_buffer_element_count(ctx) };
        let logits_buffer_element_count = unsafe { rwkv_get_logits_buffer_element_count(ctx) };
        let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();
        let _state = {
            let mut zero_state_out: Vec<f32> =
                Vec::with_capacity(state_buffer_element_count as usize);

            for _i in 0..state_buffer_element_count {
                zero_state_out.push(0.0_f32);
            }
            zero_state_out
        };
        let _logits = {
            let mut zero_state_out: Vec<f32> =
                Vec::with_capacity(logits_buffer_element_count as usize);

            for _i in 0..logits_buffer_element_count {
                zero_state_out.push(0.0_f32);
            }
            zero_state_out
        };
        Self {
            ctx,
            tokenizer,
            state_buffer_element_count,
            logits_buffer_element_count,
            model_tokens: vec![],
            model_state:  _state,
            logits: _logits,
            is_first: true
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

    pub fn rwkv_token_to_str(&self, token: &i32) -> Option<String> {
        let tokenizer = &self.tokenizer;
        tokenizer
            .decode(vec![(*token).try_into().unwrap()], false)
            .map(Some)
            .unwrap_or(None)
    }

    pub fn process_tokens(&mut self, tokens: &[u32]) {
        self.model_tokens.append(&mut tokens.to_vec());

        for token in tokens.iter() {
            self.rwkv_eval(*token as i32).unwrap();
        }
    }

    // Evaluates the given tokens with the specified configuration.
    // TODO: investigate performance of this function
    pub fn rwkv_eval(&mut self, token: i32) -> Result<(), ()> {
        let state_in = if self.is_first {
            std::ptr::null_mut()
        } else {
            self.model_state.as_mut_ptr()
        };

        let state_out = self.model_state.as_mut_ptr();
        let logits_out = self.logits.as_mut_ptr();


        let res = unsafe {
            rwkv_eval(
                self.ctx,
                token,
                state_in,
                state_out,
                logits_out,
            )
        };

        if res {
            Ok(())
        } else {
            Err(())
        }
    }
}

// Provides thread-safe behavior for RWKVContext.
unsafe impl Send for RWKVContext {}
unsafe impl Sync for RWKVContext {}

// Enables dereferencing RWKVContext to access the underlying *mut rwkv_context.
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
