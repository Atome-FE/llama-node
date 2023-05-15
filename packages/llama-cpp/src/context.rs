use std::{ffi::CStr, ptr::null_mut, slice};

use anyhow::Result;
use llama_sys::{
    llama_apply_lora_from_file, llama_context, llama_context_default_params, llama_context_params,
    llama_eval, llama_free, llama_get_embeddings, llama_get_logits, llama_init_from_file,
    llama_n_embd, llama_n_vocab, llama_print_system_info,
    llama_sample_frequency_and_presence_penalties, llama_sample_repetition_penalty,
    llama_sample_tail_free, llama_sample_temperature, llama_sample_token,
    llama_sample_token_greedy, llama_sample_top_k, llama_sample_top_p, llama_sample_typical,
    llama_token, llama_token_data, llama_token_data_array, llama_token_nl, llama_token_to_str,
};

use crate::types::{LlamaContextParams, LlamaInvocation, LlamaLoraAdaptor};

impl LlamaContextParams {
    // Returns the default parameters or the user-specified parameters.
    pub fn or_default(params: &Option<LlamaContextParams>) -> llama_context_params {
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
            n_gpu_layers: params.n_gpu_layers,
            seed: params.seed,
            f16_kv: params.f16_kv,
            logits_all: params.logits_all,
            vocab_only: params.vocab_only,
            use_mmap: params.use_mmap,
            use_mlock: params.use_mlock,
            embedding: params.embedding,
            progress_callback: None,
            progress_callback_user_data: null_mut(),
        }
    }
}

// Represents the LLamaContext which wraps FFI calls to the llama.cpp library.
pub struct LLamaContext {
    ctx: *mut llama_context,
}

impl LLamaContext {
    // Creates a new LLamaContext from the specified file and configuration parameters.
    pub async fn from_file_and_params(
        path: &str,
        params: &Option<LlamaContextParams>,
        lora_params: &Option<LlamaLoraAdaptor>,
    ) -> Self {
        let params = LlamaContextParams::or_default(params);
        let ctx = unsafe { llama_init_from_file(path.as_ptr() as *const i8, params) };
        if let Some(lora_params) = lora_params {
            let lora_base_path = lora_params
                .lora_base
                .as_ref()
                .map(|p| p.as_ptr() as *const i8)
                .unwrap_or(null_mut());

            let err = unsafe {
                llama_apply_lora_from_file(
                    ctx,
                    lora_params.lora_adapter.as_ptr() as *const i8,
                    lora_base_path,
                    lora_params.n_threads,
                )
            };

            if err != 0 {
                panic!("Failed to apply LORA adapter");
            }
        }
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
        last_n_tokens: &mut [llama_token],
        input: &LlamaInvocation,
        context_params: &llama_context_params,
    ) -> i32 {
        let n_ctx = context_params.n_ctx;
        let top_p = input.top_p.unwrap_or(0.95) as f32;
        let top_k = if input.top_k <= 0 {
            unsafe { llama_n_vocab(self.ctx) }
        } else {
            input.top_k
        };
        let tfs_z = input.tfs_z.unwrap_or(1.0) as f32;
        let temp = input.temp.unwrap_or(0.8) as f32;
        let typical_p = input.typical_p.unwrap_or(1.0) as f32;
        let repeat_penalty = input.repeat_penalty.unwrap_or(1.10) as f32;
        let repeat_last_n = input.repeat_last_n.unwrap_or(64);
        let repeat_last_n = if repeat_last_n < 0 {
            n_ctx
        } else {
            repeat_last_n
        };
        let alpha_frequency = input.frequency_penalty.unwrap_or(0.0) as f32;
        let alpha_presence = input.presence_penalty.unwrap_or(0.0) as f32;
        let penalize_nl = input.penalize_nl.unwrap_or(true);

        let n_vocab = unsafe { llama_n_vocab(self.ctx) };
        let logits_ptr = unsafe { llama_get_logits(self.ctx) };
        let logits = unsafe { slice::from_raw_parts_mut(logits_ptr, n_vocab as usize) };

        // TODO: apply logit bias

        let mut candidates: Vec<llama_token_data> = Vec::with_capacity(n_vocab as usize);

        for i in 0..n_vocab {
            candidates.push(llama_token_data {
                id: i,
                logit: logits[i as usize],
                p: 0.0_f32,
            })
        }

        let candidates_p = &mut llama_token_data_array {
            data: candidates.as_mut_ptr(),
            size: candidates.len(),
            sorted: false,
        };

        let nl_logit = logits[unsafe { llama_token_nl() } as usize];

        let last_n_repeat = std::cmp::min(
            std::cmp::min(last_n_tokens.len() as i32, repeat_last_n),
            n_ctx,
        );

        fn get_last_n_ptr(
            last_n_tokens: &mut [llama_token],
            last_n_repeat: i32,
        ) -> *mut llama_token {
            let last_n_tokens_ptr = last_n_tokens.as_ptr();
            let last_n_tokens_size = last_n_tokens.len();
            let end_ptr = unsafe { last_n_tokens_ptr.add(last_n_tokens_size) };
            unsafe { end_ptr.sub(last_n_repeat as usize) }.cast_mut()
        }

        unsafe {
            llama_sample_repetition_penalty(
                self.ctx,
                candidates_p,
                get_last_n_ptr(last_n_tokens, last_n_repeat),
                last_n_repeat as usize,
                repeat_penalty,
            );

            llama_sample_frequency_and_presence_penalties(
                self.ctx,
                candidates_p,
                get_last_n_ptr(last_n_tokens, last_n_repeat),
                last_n_repeat as usize,
                alpha_frequency,
                alpha_presence,
            )
        }

        if !penalize_nl {
            let nl = unsafe { llama_token_nl() } as usize;
            logits[nl] = nl_logit;
        }

        #[allow(unused_assignments)]
        let mut id = 0;

        if temp <= 0.0_f32 {
            id = unsafe { llama_sample_token_greedy(self.ctx, candidates_p) };
        } else {
            // TODO: here we just do temp for first approach, I dont understand microstat very well, will impl later
            id = unsafe {
                llama_sample_top_k(self.ctx, candidates_p, top_k, 1);
                llama_sample_tail_free(self.ctx, candidates_p, tfs_z, 1);
                llama_sample_typical(self.ctx, candidates_p, typical_p, 1);
                llama_sample_top_p(self.ctx, candidates_p, top_p, 1);
                llama_sample_temperature(self.ctx, candidates_p, temp);
                llama_sample_token(self.ctx, candidates_p)
            }
        }

        id
    }

    pub fn llama_token_to_str(&self, token: &i32) -> Option<String> {
        let c_ptr = unsafe { llama_token_to_str(self.ctx, *token) };
        if c_ptr.is_null() {
            return None;
        }

        let native_string = unsafe { CStr::from_ptr(c_ptr) }
            .to_string_lossy()
            .into_owned();

        Some(native_string)
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
// TODO: this is not Sync-able
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
