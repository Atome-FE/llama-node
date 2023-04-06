use std::{
  convert::Infallible,
  sync::{
    mpsc::{channel, Receiver, Sender, TryRecvError},
    Arc, Mutex,
  },
  thread,
};

use crate::types::{
  EmbeddingResult, InferenceResult, InferenceToken, LLamaCommand, LLamaConfig,
  LLamaInferenceArguments, LoadModelResult, TokenizeResult,
};
use llama_rs::{
  EvaluateOutputRequest, InferenceError, InferenceParameters, InferenceSession,
  InferenceSessionParameters, Model, ModelKVMemoryType, TokenBias, Vocabulary, EOT_TOKEN_ID,
};
use rand::SeedableRng;

#[derive(Clone)]
pub struct LLamaChannel {
  command_sender: Sender<LLamaCommand>,
  command_receiver: Arc<Mutex<Receiver<LLamaCommand>>>,
}

struct LLamaInternal {
  model: Option<Model>,
  vocab: Option<Vocabulary>,
}

fn parse_bias(s: &str) -> Result<TokenBias, String> {
  s.parse()
}

impl LLamaInternal {
  pub fn load_model(&mut self, params: &LLamaConfig, sender: &Sender<LoadModelResult>) {
    let num_ctx_tokens = params.num_ctx_tokens.unwrap_or(512);
    log::info!("num_ctx_tokens: {}", num_ctx_tokens);
    // let restore_prompt: Option<String> = None;
    // let cache_prompt: Option<String> = None;
    // let repeat_last_n = 64;
    // let num_predict = Some(128);

    if let Ok((model, vocab)) =
      llama_rs::Model::load(params.path.clone(), num_ctx_tokens as usize, |progress| {
        use llama_rs::LoadProgress;
        match progress {
          LoadProgress::HyperparametersLoaded(hparams) => {
            log::debug!("Loaded HyperParams {hparams:#?}")
          }
          LoadProgress::BadToken { index } => {
            log::info!("Warning: Bad token in vocab at index {index}")
          }
          LoadProgress::ContextSize { bytes } => log::info!(
            "ggml ctx size = {:.2} MB\n",
            bytes as f64 / (1024.0 * 1024.0)
          ),
          LoadProgress::PartLoading {
            file,
            current_part,
            total_parts,
          } => log::info!(
            "Loading model part {}/{} from '{}'\n",
            current_part,
            total_parts,
            file.to_string_lossy(),
          ),
          LoadProgress::PartTensorLoaded {
            current_tensor,
            tensor_count,
            ..
          } => {
            if current_tensor % 8 == 0 {
              log::info!("Loaded tensor {current_tensor}/{tensor_count}");
            }
          }
          LoadProgress::PartLoaded {
            file,
            byte_size,
            tensor_count,
          } => {
            log::info!("Loading of '{}' complete", file.to_string_lossy());
            log::info!(
              "Model size = {:.2} MB / num tensors = {}",
              byte_size as f64 / 1024.0 / 1024.0,
              tensor_count
            );
          }
        }
      })
    {
      self.model = Some(model);
      self.vocab = Some(vocab);

      log::info!("Model fully loaded!");

      sender
        .send(LoadModelResult {
          error: false,
          message: None,
        })
        .unwrap();
    } else {
      sender
        .send(LoadModelResult {
          error: true,
          message: Some("Could not load model".to_string()),
        })
        .unwrap();
    }
  }

  pub fn tokenize(&self, text: &str, sender: &Option<Sender<TokenizeResult>>) -> Vec<i32> {
    let vocab = self.vocab.as_ref().unwrap();
    let tokens = vocab
      .tokenize(text, false)
      .unwrap()
      .iter()
      .map(|(_, tid)| *tid)
      .collect::<Vec<_>>();
    if let Some(sender) = sender {
      sender
        .send(TokenizeResult { data: tokens.clone() })
        .unwrap();
    }
    tokens
  }

  fn get_inference_params(&self, params: &LLamaInferenceArguments) -> InferenceParameters {
    let ignore_eos = params.ignore_eos.unwrap_or(false);

    let default_token_bias = if ignore_eos {
      TokenBias::new(vec![(EOT_TOKEN_ID, -1.0)])
    } else {
      TokenBias::default()
    };

    let token_bias = if let Some(token_bias) = &params.token_bias {
      if let Ok(token_bias) = parse_bias(token_bias) {
        token_bias
      } else {
        default_token_bias
      }
    } else {
      default_token_bias
    };

    // let token_bias = params.token_bias.clone().unwrap_or_else(|| {
    //   if ignore_eos {
    //     TokenBias::new(vec![(EOD_TOKEN_ID, -1.0)])
    //   } else {
    //     TokenBias::default()
    //   }
    // });

    let n_threads = params.n_threads.unwrap_or(4) as usize;
    let n_batch = params.n_batch.unwrap_or(8) as usize;
    let top_k = params.top_k.unwrap_or(30) as usize;
    let temperature = params.temp.unwrap_or(0.8) as f32;

    let inference_params = InferenceParameters {
      n_threads,
      n_batch,
      top_k,
      top_p: params.top_p.unwrap_or(0.95) as f32,
      repeat_penalty: params.repeat_penalty.unwrap_or(1.30) as f32,
      temperature,
      bias_tokens: token_bias,
      play_back_previous_tokens: false,
      ..Default::default()
    };

    // log::info!("repeat_last_n: {}", repeat_last_n);
    log::info!("n_threads: {}", inference_params.n_threads);
    log::info!("n_batch: {}", inference_params.n_batch);
    log::info!("top_k: {}", inference_params.top_k);
    log::info!("top_p: {}", inference_params.top_p);
    log::info!("repeat_penalty: {}", inference_params.repeat_penalty);
    log::info!("temp: {}", inference_params.temperature);
    // log::info!("seed: {:?}", seed);

    inference_params
  }

  fn start_new_session(&self, params: &LLamaInferenceArguments) -> InferenceSession {
    let model = self.model.as_ref().unwrap();
    let repeat_last_n = params.repeat_last_n.unwrap_or(512) as usize;
    let float16 = params.float16.unwrap_or(false);

    let inference_session_params = {
      let mem_typ = if float16 {
        ModelKVMemoryType::Float16
      } else {
        ModelKVMemoryType::Float32
      };
      InferenceSessionParameters {
        memory_k_type: mem_typ,
        memory_v_type: mem_typ,
        repetition_penalty_last_n: repeat_last_n,
      }
    };

    model.start_session(inference_session_params)
  }

  pub fn get_word_embedding(
    &self,
    params: &LLamaInferenceArguments,
    sender: &Sender<EmbeddingResult>,
  ) {
    let mut session = self.start_new_session(params);
    let inference_params = self.get_inference_params(params);
    let model = self.model.as_ref().unwrap();
    let vocab = self.vocab.as_ref().unwrap();
    let prompt_for_feed = format!(" {}", params.prompt);

    if let Err(InferenceError::ContextFull) =
      session.feed_prompt::<Infallible>(model, vocab, &inference_params, prompt_for_feed.as_str(), |_| {
        Ok(())
      })
    {
      sender
        .send(EmbeddingResult::EmbeddingError(
          "Context window full.".to_string(),
        ))
        .unwrap();
    }

    let end_token = self.tokenize("\n", &None);

    let mut output_request = EvaluateOutputRequest {
      all_logits: None,
      embeddings: Some(Vec::new()),
    };

    model.evaluate(
      &mut session,
      &inference_params,
      &end_token,
      &mut output_request,
    );

    sender
      .send(EmbeddingResult::EmbeddingData(output_request.embeddings))
      .unwrap();
  }

  pub fn inference(&mut self, params: &LLamaInferenceArguments, sender: &Sender<InferenceResult>) {
    let num_predict = params.num_predict.unwrap_or(512) as usize;
    let model = self.model.as_ref().unwrap();
    let vocab = self.vocab.as_ref().unwrap();

    let prompt = &params.prompt;
    let feed_prompt = params.feed_prompt.unwrap_or(false);
    let seed = params.seed.map(|seed| seed as u64);

    let mut session = self.start_new_session(params);
    let inference_params = self.get_inference_params(params);

    let mut rng = if let Some(seed) = seed {
      rand::rngs::StdRng::seed_from_u64(seed)
    } else {
      rand::rngs::StdRng::from_entropy()
    };

    if let Err(InferenceError::ContextFull) =
      session.feed_prompt::<Infallible>(model, vocab, &inference_params, prompt, |_| Ok(()))
    {
      sender
        .send(InferenceResult::InferenceError(
          "Context window full.".to_string(),
        ))
        .unwrap();
    }

    let inference_input = if feed_prompt { "" } else { prompt };

    let res = session.inference_with_prompt::<Infallible>(
      model,
      vocab,
      &inference_params,
      inference_input,
      Some(num_predict),
      &mut rng,
      |t| {
        let to_send = InferenceResult::InferenceData(InferenceToken {
          token: t.to_string(),
          completed: false,
        });

        sender.send(to_send).unwrap();

        Ok(())
      },
    );

    match res {
      Ok(_) => {
        let to_send = InferenceResult::InferenceData(InferenceToken {
          token: "\n\n<end>\n".to_string(),
          completed: true,
        });
        sender.send(to_send).unwrap();
      }
      Err(llama_rs::InferenceError::EndOfText) => {
        let to_send = InferenceResult::InferenceError("End of text.".to_string());
        sender.send(to_send).unwrap();
      }
      Err(llama_rs::InferenceError::ContextFull) => {
        sender
          .send(InferenceResult::InferenceError(
            "Context window full, stopping inference.".to_string(),
          ))
          .unwrap();
        log::warn!("Context window full, stopping inference.")
      }
      Err(llama_rs::InferenceError::TokenizationFailed) => {
        sender
          .send(InferenceResult::InferenceError(
            "Failed to tokenize initial prompt.".to_string(),
          ))
          .unwrap();
      }
      Err(llama_rs::InferenceError::UserCallback(_)) => unreachable!("cannot fail"),
    }
    sender.send(InferenceResult::InferenceEnd).unwrap();
  }
}

impl LLamaChannel {
  pub fn new() -> Arc<Self> {
    let (command_sender, command_receiver) = channel::<LLamaCommand>();

    let channel = LLamaChannel {
      command_receiver: Arc::new(Mutex::new(command_receiver)),
      command_sender,
    };

    channel.spawn();

    Arc::new(channel)
  }

  pub fn load_model(&self, params: LLamaConfig, sender: Sender<LoadModelResult>) {
    self
      .command_sender
      .send(LLamaCommand::LoadModel(params, sender))
      .unwrap();
  }

  pub fn inference(&self, params: LLamaInferenceArguments, sender: Sender<InferenceResult>) {
    self
      .command_sender
      .send(LLamaCommand::Inference(params, sender))
      .unwrap();
  }

  pub fn get_word_embedding(
    &self,
    params: LLamaInferenceArguments,
    sender: Sender<EmbeddingResult>,
  ) {
    self
      .command_sender
      .send(LLamaCommand::Embedding(params, sender))
      .unwrap()
  }

  pub fn tokenize(&self, text: &str, sender: Sender<TokenizeResult>) {
    self
      .command_sender
      .send(LLamaCommand::Tokenize(text.to_string(), sender))
      .unwrap();
  }

  // llama instance main loop
  pub fn spawn(&self) {
    let rv = self.command_receiver.clone();

    thread::spawn(move || {
      let mut llama = LLamaInternal {
        model: None,
        vocab: None,
      };

      let rv = rv.lock().unwrap();

      'llama_loop: loop {
        let command = rv.try_recv();
        match command {
          Ok(LLamaCommand::Inference(params, sender)) => {
            llama.inference(&params, &sender);
          }
          Ok(LLamaCommand::LoadModel(params, sender)) => {
            llama.load_model(&params, &sender);
          }
          Ok(LLamaCommand::Embedding(params, sender)) => {
            llama.get_word_embedding(&params, &sender);
          }
          Ok(LLamaCommand::Tokenize(text, sender)) => {
            llama.tokenize(&text, &Some(sender));
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
