use std::{
    sync::{
        mpsc::{channel, Receiver, Sender, TryRecvError},
        Arc, Mutex,
    },
    thread,
};

use crate::{
    context::{LLamaContext, LlamaContextParams, LlamaInvocation},
    tokenizer::{embedding_to_output, llama_token_eos, tokenize},
};

#[derive(Clone)]
pub struct LLamaChannel {
    command_sender: Sender<LLamaCommand>,
    command_receiver: Arc<Mutex<Receiver<LLamaCommand>>>,
}

pub struct LLamaInternal {
    context: LLamaContext,
    context_params: Option<LlamaContextParams>,
}

#[derive(Clone, Debug)]
pub enum LLamaCommand {
    Inference(LlamaInvocation, Sender<String>),
}

impl LLamaInternal {
    pub fn inference(&self, input: &LlamaInvocation, sender: &Sender<String>) {
        let context_params_c = LlamaContextParams::or_default(&self.context_params);
        log::info!("inference: {:?}", input);
        log::info!("context_params: {:?}", context_params_c);
        let input_ctx = &self.context;
        // Tokenize the stop sequence and input prompt.
        let tokenized_stop_prompt = if let Some(stop_sequence) = &input.stop_sequence {
            Some(
                tokenize(
                    input_ctx,
                    &stop_sequence,
                    context_params_c.n_ctx as usize,
                    false,
                )
                .unwrap(),
            )
        } else {
            None
        };

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

        log::info!("hard coded token_eos: {}", token_eos);

        // Generate remaining tokens.
        let mut n_remaining = context_params_c.n_ctx - tokenized_input.len() as i32;
        let mut n_used = tokenized_input.len() - 1;
        let mut stop_sequence_i = 0;
        while n_remaining > 0 {
            let tok = input_ctx.llama_sample(embd.as_slice(), n_used as i32, input);
            n_used += 1;
            n_remaining -= 1;
            embd[n_used] = tok;
            if tok == token_eos {
                break;
            }
            if input.n_tok_predict != 0
                && n_used > (input.n_tok_predict as usize) + tokenized_input.len() - 1
            {
                break;
            }

            if let Some(tokenized_stop_prompt) = &tokenized_stop_prompt {
                if tok == tokenized_stop_prompt[stop_sequence_i] {
                    stop_sequence_i += 1;
                    if stop_sequence_i >= tokenized_stop_prompt.len() {
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

            if stop_sequence_i == 0 {
                sender.send(output).unwrap();
            }
        }
        embedding_to_output(
            input_ctx,
            &embd[tokenized_input.len()..n_used + 1 - stop_sequence_i],
        );
    }
}

impl LLamaChannel {
    pub fn new(
        path: String,
        params: Option<LlamaContextParams>,
        load_result_sender: Sender<bool>,
        enable_logger: bool,
    ) -> Arc<Self> {
        let (command_sender, command_receiver) = channel::<LLamaCommand>();

        let channel = LLamaChannel {
            command_receiver: Arc::new(Mutex::new(command_receiver)),
            command_sender,
        };

        channel.spawn(path, params, load_result_sender, enable_logger);

        Arc::new(channel)
    }

    pub fn inference(&self, params: LlamaInvocation, sender: Sender<String>) {
        self.command_sender
            .send(LLamaCommand::Inference(params, sender))
            .unwrap();
    }

    // llama instance main loop
    pub fn spawn(
        &self,
        path: String,
        params: Option<LlamaContextParams>,
        load_result_sender: Sender<bool>,
        enable_logger: bool,
    ) {
        let rv = self.command_receiver.clone();

        thread::spawn(move || {
            let llama = LLamaInternal {
                context: LLamaContext::from_file_and_params(&path, &params),
                context_params: params,
            };

            if enable_logger {
                llama.context.llama_print_system_info();
            }

            load_result_sender.send(true).unwrap();

            let rv = rv.lock().unwrap();

            'llama_loop: loop {
                let command = rv.try_recv();
                match command {
                    Ok(LLamaCommand::Inference(params, sender)) => {
                        llama.inference(&params, &sender);
                    }
                    Err(TryRecvError::Disconnected) => {
                        break 'llama_loop;
                    }
                    _ => {
                        thread::yield_now();
                    }
                }
            }
        });
    }
}
