use std::ffi::CStr;
use std::fs::File;
use std::io::{Read, Write};
use tokenizers::tokenizer::Tokenizer;

use anyhow::Result;
use serde::{Deserialize, Serialize};
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
    pub end_string: Option<String>,
    pub seed: Option<i32>,
    pub prompt: String,
    pub is_skip_generation: Option<bool>,
    pub session_file_path: Option<String>,
    pub is_overwrite_session_file: Option<bool>,
    pub presence_penalty: Option<f64>,
    pub frequency_penalty: Option<f64>
}

// Represents the RWKVContext which wraps FFI calls to the rwkv.cpp library.
#[derive(Clone)]
pub struct RWKVContext {
    ctx: *mut rwkv_context,
    pub tokenizer: Tokenizer,
}

#[derive(Clone)]
pub struct RWKVSession<'a> {
    rwkv_context: &'a RWKVContext,
    state_buffer_element_count: usize,
    logits_buffer_element_count: usize,
    model_tokens: Vec<i32>,
    model_state: Vec<f32>,
    pub logits: Vec<f32>,
    is_first: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RWKVSessionData {
    state_buffer_element_count: usize,
    logits_buffer_element_count: usize,
    model_state: Vec<f32>,
    pub logits: Vec<f32>,
    is_first: bool,
}

impl<'a> RWKVSession<'a> {
    pub fn save_to_file(&mut self, path: &str) {
        let data = RWKVSessionData {
            state_buffer_element_count: self.state_buffer_element_count,
            logits_buffer_element_count: self.logits_buffer_element_count,
            model_state: self.model_state.to_vec(),
            logits: self.logits.to_vec(),
            is_first: self.is_first
        };

        let mut file = File::create(path).expect("Could not create file");
        let serialized = bincode::serialize(&data).expect("Could not serialize session data");
        file.write_all(&serialized).expect("Could not write session data");
        log::info!("success on save to {}", path);
    }

    pub fn load_from_file_or_create(path: &str, rwkv_context: &'a RWKVContext) -> RWKVSession<'a> {
        match File::open(path) {
            Ok(mut file) => {
                let mut buffer = Vec::new();
                file.read_to_end(&mut buffer).expect("Could not read session data");
                let data: RWKVSessionData = bincode::deserialize(&buffer).expect("Could not deserialize session data");
                log::info!("success on load {}", path);
                RWKVSession{
                    rwkv_context,
                    state_buffer_element_count: data.state_buffer_element_count,
                    logits_buffer_element_count: data.logits_buffer_element_count,
                    model_tokens: Vec::new(),
                    model_state: data.model_state,
                    logits: data.logits,
                    is_first: data.is_first
                }

            }
            Err(_e) => {
                rwkv_context.create_new_session()
            }
        }
    }

    pub fn process_tokens(&mut self, tokens: &[i32]) {
        self.model_tokens.append(&mut tokens.to_vec());

        for token in tokens.iter() {
            self.rwkv_eval(*token).unwrap();
        }
    }

    // Evaluates the given tokens with the specified configuration.
    pub fn rwkv_eval(&mut self, token: i32) -> Result<(), ()> {
        let state_in = if self.is_first {
            self.is_first = false;
            std::ptr::null_mut()
        } else {
            self.model_state.as_mut_ptr()
        };

        let state_out = self.model_state.as_mut_ptr();
        let logits_out = self.logits.as_mut_ptr();

        assert!(
            self.model_state.len() == self.state_buffer_element_count,
            "state buffer size mismatch"
        );

        assert!(
            self.logits.len() == self.logits_buffer_element_count,
            "logits buffer size mismatch"
        );

        let res = unsafe {
            rwkv_eval(
                self.rwkv_context.ctx,
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

impl RWKVContext {
    // Creates a new RWKVContext from the specified file and configuration parameters.
    pub fn new(model_path: &str, tokenizer_path: &str, n_threads: u32) -> Self {
        let ctx = unsafe { rwkv_init_from_file(model_path.as_ptr() as *const i8, n_threads) };
        let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();

        Self { ctx, tokenizer }
    }

    pub fn rwkv_print_system_info_string(&self) {
        let sys_info_c_str = unsafe { rwkv_get_system_info_string() };
        let sys_info = unsafe { CStr::from_ptr(sys_info_c_str) }
            .to_str()
            .unwrap()
            .to_owned();
        log::info!("{}", sys_info);
    }

    pub fn rwkv_tokens_to_str(&self, tokens: &[u32]) -> Option<String> {
        let tokenizer = &self.tokenizer;
        tokenizer
            .decode(tokens.to_vec(), false)
            .map(Some)
            .unwrap_or(None)
    }

    pub fn create_new_session(&self) -> RWKVSession {
        let state_buffer_element_count =
            unsafe { rwkv_get_state_buffer_element_count(self.ctx) } as usize;
        let logits_buffer_element_count =
            unsafe { rwkv_get_logits_buffer_element_count(self.ctx) } as usize;

        let state: Vec<f32> = vec![0.0_f32; state_buffer_element_count];
        let zero_logits_out: Vec<f32> = vec![0.0_f32; logits_buffer_element_count];

        RWKVSession {
            rwkv_context: self,
            state_buffer_element_count,
            logits_buffer_element_count,
            model_tokens: vec![],
            model_state: state,
            logits: zero_logits_out,
            is_first: true,
        }
    }
}

// Provides thread-safe behavior for RWKVContext.
unsafe impl Send for RWKVContext {}
// TODO: this is not Sync-able
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
