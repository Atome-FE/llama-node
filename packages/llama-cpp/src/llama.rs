use std::sync::Arc;

use anyhow::Result;
use tokio::sync::Mutex;

use crate::{
    context::LLamaContext,
    tokenizer::{llama_token_eos, tokenize},
    types::{InferenceResult, InferenceResultType, InferenceToken, Generate, ModelLoad},
};

pub struct LLamaInternal {
    context: LLamaContext,
    context_params: ModelLoad,
}

impl LLamaInternal {
    pub async fn load(
        params: ModelLoad,
        enable_logger: bool,
    ) -> Result<Arc<Mutex<LLamaInternal>>, napi::Error> {
        let llama = LLamaInternal {
            context: LLamaContext::from_file_and_params(&params).await?,
            context_params: params,
        };

        if enable_logger {
            llama.context.llama_print_system_info().map_err(|e| {
                napi::Error::from_reason(format!("Failed to print system info: {:?}", e))
            })?;
        }

        Ok(Arc::new(Mutex::new(llama)))
    }
    pub async fn tokenize(&self, input: &str) -> Result<Vec<i32>, napi::Error> {
        let context = &self.context;
        Ok(tokenize(context, input, false))
    }

    pub async fn embedding(&self, input: &Generate) -> Result<Vec<f64>, napi::Error> {
        let context = &self.context;
        let embd_inp = tokenize(context, input.prompt.as_str(), true);

        // let end_text = "\n";
        // let end_token =
        //     tokenize(input_ctx, end_text, context_params_c.n_ctx as usize, false).unwrap();

        context
            .llama_eval(embd_inp.as_slice(), embd_inp.len() as i32, 0, input)
            .map_err(|e| napi::Error::from_reason(format!("Failed to evaluate input: {:?}", e)))?;

        let embeddings = context.llama_get_embeddings();

        if let Ok(embeddings) = embeddings {
            Ok(embeddings.iter().map(|&x| x as f64).collect())
        } else {
            Err(napi::Error::from_reason("Failed to get embeddings"))
        }
    }

    pub fn inference(
        &self,
        input: &Generate,
        running: Arc<Mutex<bool>>,
        callback: impl Fn(InferenceResult),
    ) -> Result<(), napi::Error> {
        let context = &self.context;
        let context_params_c = ModelLoad::to_llama_context_params(&self.context_params);
        // Tokenize the stop sequence and input prompt.
        let tokenized_stop_prompt = input
            .stop_sequence
            .as_ref()
            .map(|stop_sequence| tokenize(context, stop_sequence, false));

        log::info!("tokenized_stop_prompt: {:?}", tokenized_stop_prompt);

        let tokenized_input = tokenize(context, input.prompt.as_str(), true);

        // Embd contains the prompt and the completion. The longer the prompt, the shorter the completion.
        let mut embd = tokenized_input.clone();
        embd.resize(context_params_c.n_ctx as usize, 0);

        // Feed prompt to the model.
        context.llama_eval(
            tokenized_input.as_slice(),
            tokenized_input.len() as i32,
            0,
            input,
        )?;
        let token_eos = llama_token_eos();

        // Generate remaining tokens.
        let mut n_remaining = context_params_c.n_ctx - tokenized_input.len() as i32;
        let mut n_used = tokenized_input.len() - 1;
        let mut stop_sequence_i = 0;
        let mut completed = false;

        while n_remaining > 0 {
            // Check if we are aborted by caller.
            let running = *running.blocking_lock();
            if !running {
                break;
            }

            n_used += 1;
            n_remaining -= 1;

            let tok = context.llama_sample(embd.as_mut_slice(), input, &context_params_c);
            embd[n_used] = tok;

            if tok == token_eos {
                completed = true;
                break;
            }

            // If we are predicting a fixed number of tokens, check if we have reached that number.
            if input.n_tok_predict != 0
                && n_used > (input.n_tok_predict as usize) + tokenized_input.len() - 1
            {
                return Err(napi::Error::from_reason("Too many tokens predicted"));
            }

            // Check if we have reached the stop sequence.
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

            // We can output the token.
            let output = context.llama_token_to_str(&embd[n_used]);
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

            // Continue feeding the token to the model.
            context.llama_eval(&embd[n_used..], 1, n_used as i32, input)?;
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

        Ok(())
    }
}
