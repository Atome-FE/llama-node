use std::collections::HashMap;
use std::ops::Add;
use std::sync::Arc;

use tokio::sync::Mutex;

use crate::{
    context::{RWKVContext, RWKVInvocation},
    sampling::sample_logits,
    types::{InferenceResult, InferenceResultType, InferenceToken},
};
use crate::context::RWKVSession;

#[derive(Clone)]
pub struct RWKVInternal {
    context: RWKVContext,
}

impl RWKVInternal {
    pub async fn load(
        mode_path: String,
        tokenizer_path: String,
        n_threads: u32,
        enable_logger: bool,
    ) -> Arc<Mutex<Self>> {
        let rwkv = RWKVInternal {
            context: RWKVContext::new(&mode_path, &tokenizer_path, n_threads),
        };

        if enable_logger {
            rwkv.context.rwkv_print_system_info_string();
        }

        Arc::new(Mutex::new(rwkv))
    }
    pub async fn tokenize(&self, input: &str) -> Result<Vec<i32>, napi::Error> {
        let tokenizer = &self.context.tokenizer;
        let tokens_result = tokenizer.encode(input, false).map(Some).unwrap_or(None);
        if let Some(result) = tokens_result {
            let tokens = result.get_ids().to_vec();
            Ok(tokens.iter().map(|x| *x as i32).collect())
        } else {
            Err(napi::Error::from_reason("Failed to tokenize"))
        }
    }

    pub fn inference(
        &mut self,
        input: &RWKVInvocation,
        running: Arc<Mutex<bool>>,
        callback: impl Fn(InferenceResult),
    ) {
        let end_token = input.end_token.unwrap_or(0) as usize;
        let end_string = &input.end_string;
        let temp = input.temp as f32;
        let top_p = input.top_p as f32;
        let seed = input.seed.map(|x| x as u64);

        let context = &mut self.context;
        let tokenizer = &context.tokenizer;
        let prompt = &input.prompt;
        let binding = tokenizer.encode(prompt.as_str(), false).unwrap();
        let tokens = binding
            .get_ids()
            .iter()
            .map(|x| *x as i32)
            .collect::<Vec<i32>>();


        let session_file_path = &input.session_file_path;
        let is_overwrite_session_file = &input.is_overwrite_session_file.unwrap_or(false);
        let is_skip_generation = &input.is_skip_generation.unwrap_or(false);
        let presence_penalty = input.presence_penalty.unwrap_or(0.2f64) as f32;
        let frequency_penalty = input.frequency_penalty.unwrap_or(0.2f64) as f32;

        let mut session = match session_file_path {
            Some(path) =>
                RWKVSession::load_from_file_or_create(path.as_str(), context),
            None =>
                context.create_new_session()
        };

        session.process_tokens(&tokens);

        let mut accumulated_token: Vec<u32> = Vec::new();
        let mut generated_string: String = String::new();
        let mut token_counts: HashMap<u32, u32> = HashMap::new();

        for _i in 0..input.max_predict_length {
            // Check if we are aborted by caller.
            let running = *running.blocking_lock();
            if !running {
                break;
            }

            let logits: &mut [f32] = session.logits.as_mut();

            for (k, v) in &token_counts {
                let penalty: f32 = presence_penalty + (*v as f32) * frequency_penalty;
                logits[*k as usize] -= penalty
            }

            let token = sample_logits(logits, temp, top_p, &seed);
            let token_u32 = token as u32;

            match token_counts.get(&token_u32) {
                Some(v) => {
                    token_counts.insert(token_u32, v + 1);
                }
                None => {
                    token_counts.insert(token_u32, 1);
                }
            }

            accumulated_token.push(token_u32);

            let decoded = context.rwkv_tokens_to_str(&accumulated_token).unwrap();
            generated_string = generated_string.add(decoded.as_str());

            let is_match_end_string: bool = match end_string {
                Some(str) => {
                    generated_string.contains(str)
                }
                None => {
                    false
                }
            };

            if token >= 50276 || token == end_token || is_match_end_string || *is_skip_generation {
                callback(InferenceResult {
                    r#type: InferenceResultType::Data,
                    message: None,
                    data: Some(InferenceToken {
                        token: "\n\n<end>\n".to_string(),
                        completed: true,
                    }),
                });
                break;
            }

            if !decoded.contains('\u{FFFD}') {
                accumulated_token.clear();
                callback(InferenceResult {
                    r#type: InferenceResultType::Data,
                    message: None,
                    data: Some(InferenceToken {
                        token: decoded,
                        completed: false,
                    }),
                });
            }

            session.process_tokens(&[token.try_into().unwrap()]);
        }

        match session_file_path {
            Some(path) =>
                if *is_overwrite_session_file {
                    session.save_to_file(path.as_str());
                }
            None => {}
        };

        callback(InferenceResult {
            r#type: InferenceResultType::End,
            message: None,
            data: None,
        });
    }
}
