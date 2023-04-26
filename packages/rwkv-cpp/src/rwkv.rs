use std::{
    io::Write,
    sync::{
        mpsc::{channel, Receiver, Sender, TryRecvError},
        Arc, Mutex,
    },
    thread,
};

use crate::{
    context::{RWKVContext, RWKVInvocation},
    types::{
        EmbeddingResult, InferenceResult, InferenceResultType, InferenceToken, RWKVCommand,
        TokenizeResult, TokenizeResultType,
    },
};

#[derive(Clone)]
pub struct RWKVChannel {
    command_sender: Sender<RWKVCommand>,
    command_receiver: Arc<Mutex<Receiver<RWKVCommand>>>,
}

#[derive(Clone)]
pub struct RWKVInternal {
    context: RWKVContext,
}

impl RWKVInternal {
    pub fn tokenize(&self, input: &str, sender: &Sender<TokenizeResult>) {
        let tokenizer = &self.context.tokenizer;
        let tokens_result = tokenizer.encode(input, true).map(Some).unwrap_or(None);
        if let Some(result) = tokens_result {
            let tokens = result.get_ids().to_vec();
            sender
                .send(TokenizeResult {
                    r#type: TokenizeResultType::Data,
                    data: tokens.iter().map(|x| *x as i32).collect(),
                })
                .unwrap();
        } else {
            sender
                .send(TokenizeResult {
                    r#type: TokenizeResultType::Error,
                    data: vec![],
                })
                .unwrap();
        }
    }

    /* pub fn embedding(&self, input: &LlamaInvocation, sender: &Sender<EmbeddingResult>) {
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
            sender
                .send(EmbeddingResult {
                    r#type: EmbeddingResultType::Data,
                    data: embeddings.iter().map(|&x| x as f64).collect(),
                })
                .unwrap();
        } else {
            sender
                .send(EmbeddingResult {
                    r#type: EmbeddingResultType::Error,
                    data: vec![],
                })
                .unwrap();
        }
    } */

    pub fn inference(&mut self, input: &RWKVInvocation, sender: &Sender<InferenceResult>) {
        let context = &mut self.context;
        let tokenizer = &context.tokenizer;
        let prompt = &input.prompt;
        let binding = tokenizer.encode(prompt.as_str(), false).unwrap();
        let tokens = binding.get_ids();
        context.process_tokens(tokens);

        for _i in 0..256 {
            let mut logits = context.logits.clone().unwrap();
            let token =
                crate::sampling::sample_logits(&mut logits, input.temp as f32, input.top_p as f32);
            if token >= 50276 {
                break;
            }

            let decoded = context.rwkv_token_to_str(&(token as i32)).unwrap();
            print!("{}", decoded);
            std::io::stdout().flush().unwrap();

            sender
                .send(InferenceResult {
                    r#type: InferenceResultType::Data,
                    message: None,
                    data: Some(InferenceToken {
                        token: decoded,
                        completed: false,
                    }),
                })
                .unwrap();

            context.process_tokens(&[token.try_into().unwrap()]);
            // println!("sent");
        }
        // context.rwkv_token_to_str(token)

        // thread::sleep(std::time::Duration::from_millis(10000));
    }
}

impl RWKVChannel {
    pub fn new(
        model_path: String,
        tokenizer_path: String,
        params: u32,
        load_result_sender: Sender<bool>,
        enable_logger: bool,
    ) -> Arc<Self> {
        let (command_sender, command_receiver) = channel::<RWKVCommand>();

        let channel = RWKVChannel {
            command_receiver: Arc::new(Mutex::new(command_receiver)),
            command_sender,
        };

        channel.spawn(
            model_path,
            tokenizer_path,
            params,
            load_result_sender,
            enable_logger,
        );

        Arc::new(channel)
    }

    pub fn tokenize(&self, input: String, sender: Sender<TokenizeResult>) {
        self.command_sender
            .send(RWKVCommand::Tokenize(input, sender))
            .unwrap();
    }

    pub fn embedding(&self, params: RWKVInvocation, sender: Sender<EmbeddingResult>) {
        self.command_sender
            .send(RWKVCommand::Embedding(params, sender))
            .unwrap();
    }

    pub fn inference(&self, params: RWKVInvocation, sender: Sender<InferenceResult>) {
        self.command_sender
            .send(RWKVCommand::Inference(params, sender))
            .unwrap();
    }

    // rwkv instance main loop
    pub fn spawn(
        &self,
        mode_path: String,
        tokenizer_path: String,
        params: u32,
        load_result_sender: Sender<bool>,
        enable_logger: bool,
    ) {
        let rv = self.command_receiver.clone();

        thread::spawn(move || {
            let mut rwkv = RWKVInternal {
                context: RWKVContext::from_file_and_params(&mode_path, &tokenizer_path, params),
            };

            if enable_logger {
                rwkv.context.rwkv_print_system_info_string();
            }

            load_result_sender.send(true).unwrap();

            let rv = rv.lock().unwrap();

            'rwkv_loop: loop {
                let command = rv.try_recv();
                match command {
                    Ok(RWKVCommand::Inference(params, sender)) => {
                        rwkv.inference(&params, &sender);
                    }
                    // Ok(RWKVCommand::Embedding(params, sender)) => {
                    //     rwkv.embedding(&params, &sender);
                    // }
                    Ok(RWKVCommand::Tokenize(text, sender)) => {
                        rwkv.tokenize(&text, &sender);
                    }
                    Err(TryRecvError::Disconnected) => {
                        break 'rwkv_loop;
                    }
                    _ => {
                        thread::yield_now();
                    }
                }
            }
        });
    }
}
