use std::sync::Arc;

use crate::{
    context::{LLamaContext},
    tokenizer::{llama_token_eos, tokenize},
    types::{InferenceResult, InferenceResultType, InferenceToken, LlamaContextParams, LlamaInvocation},
};

#[derive(Clone)]
pub struct LLamaInternal {
    context: LLamaContext,
    context_params: Option<LlamaContextParams>,
}

impl LLamaInternal {
    pub async fn load(
        path: String,
        params: Option<LlamaContextParams>,
        enable_logger: bool,
    ) -> Arc<Self> {
        let llama = LLamaInternal {
            context: LLamaContext::from_file_and_params(&path, &params).await,
            context_params: params,
        };

        if enable_logger {
            llama.context.llama_print_system_info();
        }

        Arc::new(llama)
    }
    pub async fn tokenize(&self, input: &str, n_ctx: usize) -> Result<Vec<i32>, napi::Error> {
        if let Ok(data) = tokenize(&self.context, input, n_ctx, false) {
            Ok(data)
        } else {
            Err(napi::Error::from_reason("Failed to tokenize"))
        }
    }

    pub async fn embedding(&self, input: &LlamaInvocation) -> Result<Vec<f64>, napi::Error> {
        let context_params_c = LlamaContextParams::or_default(&self.context_params);
        let input_ctx = &self.context;
        let embd_inp = tokenize(
            input_ctx,
            input.prompt.as_str(),
            context_params_c.n_ctx as usize,
            true,
        )
        .unwrap();

        // let end_text = "\n";
        // let end_token =
        //     tokenize(input_ctx, end_text, context_params_c.n_ctx as usize, false).unwrap();

        input_ctx
            .llama_eval(embd_inp.as_slice(), embd_inp.len() as i32, 0, input)
            .unwrap();

        let embeddings = input_ctx.llama_get_embeddings();

        if let Ok(embeddings) = embeddings {
            Ok(embeddings.iter().map(|&x| x as f64).collect())
        } else {
            Err(napi::Error::from_reason("Failed to get embeddings"))
        }
    }

    pub async fn inference(&self, input: &LlamaInvocation, callback: impl Fn(InferenceResult)) {
        let context_params_c = LlamaContextParams::or_default(&self.context_params);
        let input_ctx = &self.context;
        // Tokenize the stop sequence and input prompt.
        let tokenized_stop_prompt = input.stop_sequence.as_ref().map(|stop_sequence| {
            tokenize(
                input_ctx,
                stop_sequence,
                context_params_c.n_ctx as usize,
                false,
            )
            .unwrap()
        });

        log::info!("tokenized_stop_prompt: {:?}", tokenized_stop_prompt);

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
                input,
            )
            .unwrap();
        let token_eos = llama_token_eos();

        // Generate remaining tokens.
        let mut n_remaining = context_params_c.n_ctx - tokenized_input.len() as i32;
        let mut n_used = tokenized_input.len() - 1;
        let mut stop_sequence_i = 0;
        let mut completed = false;
        while n_remaining > 0 {
            let tok = input_ctx.llama_sample(embd.as_mut_slice(), input, &context_params_c);
            n_used += 1;
            n_remaining -= 1;
            embd[n_used] = tok;
            if tok == token_eos {
                completed = true;
                break;
            }
            if input.n_tok_predict != 0
                && n_used > (input.n_tok_predict as usize) + tokenized_input.len() - 1
            {
                callback(InferenceResult {
                    r#type: InferenceResultType::Error,
                    data: None,
                    message: Some("Too many tokens predicted".to_string()),
                });
                break;
            }

            if let Some(tokenized_stop_prompt) = &tokenized_stop_prompt {
                if tok == tokenized_stop_prompt[stop_sequence_i] {
                    stop_sequence_i += 1;
                    if stop_sequence_i >= tokenized_stop_prompt.len() {
                        completed = true;
                        break;
                    }
                } else {
                    stop_sequence_i = 0;
                }
            }

            input_ctx
                .llama_eval(&embd[n_used..], 1, n_used as i32, input)
                .unwrap();

            let output = input_ctx.llama_token_to_str(&embd[n_used]);

            if let Some(output) = output {
                if stop_sequence_i == 0 {
                    callback(InferenceResult {
                        r#type: InferenceResultType::Data,
                        data: Some(InferenceToken {
                            token: output,
                            completed: false,
                        }),
                        message: None,
                    });
                }
            }
        }

        if completed {
            callback(InferenceResult {
                r#type: InferenceResultType::Data,
                data: Some(InferenceToken {
                    token: "\n\n<end>\n".to_string(),
                    completed: true,
                }),
                message: None,
            });
        }

        callback(InferenceResult {
            r#type: InferenceResultType::End,
            data: None,
            message: None,
        });
    }
}
