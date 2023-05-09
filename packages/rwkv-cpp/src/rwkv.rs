use std::sync::Arc;

use tokio::sync::Mutex;

use crate::{
    context::{RWKVContext, RWKVInvocation},
    sampling::sample_logits,
    types::{InferenceResult, InferenceResultType, InferenceToken},
};

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

    pub async fn inference(&mut self, input: &RWKVInvocation, callback: impl Fn(InferenceResult)) {
        let end_token = input.end_token.unwrap_or(0) as usize;
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

        let mut session = context.create_new_session();

        session.process_tokens(&tokens);

        let mut accumulated_token: Vec<u32> = Vec::new();

        for _i in 0..input.max_predict_length {
            let logits = session.logits.as_mut();
            let token = sample_logits(logits, temp, top_p, &seed);

            if token >= 50276 || token == end_token {
                callback(InferenceResult {
                    r#type: InferenceResultType::Data,
                    message: None,
                    data: Some(InferenceToken {
                        token: "\n\n<end>\n".to_string(),
                        completed: true,
                    }),
                });
                return;
            }

            accumulated_token.push(token as u32);

            let decoded = context.rwkv_tokens_to_str(&accumulated_token).unwrap();

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
    }
}
