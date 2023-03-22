use std::{
  convert::Infallible,
  sync::{
    mpsc::{channel, Receiver, Sender, TryRecvError},
    Arc, Mutex,
  },
  thread,
};

use crate::types::{
  InferenceResult, InferenceToken, LLamaArguments, LLamaCommand, LLamaConfig, LoadModelResult,
};
use llama_rs::{InferenceParameters, Model, OutputToken, Vocabulary};
use napi::bindgen_prelude::BigInt;
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

impl LLamaInternal {
  pub fn load_model(&mut self, params: LLamaConfig, sender: Sender<LoadModelResult>) {
    let num_ctx_tokens = params.num_ctx_tokens.unwrap_or(512);
    log::info!("num_ctx_tokens: {}", num_ctx_tokens);
    // let restore_prompt: Option<String> = None;
    // let cache_prompt: Option<String> = None;
    // let repeat_last_n = 64;
    // let num_predict = Some(128);

    let sender = sender.clone();

    if let Ok((model, vocab)) = llama_rs::Model::load(params.path, num_ctx_tokens, |progress| {
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
        LoadProgress::MemorySize { bytes, n_mem } => log::info!(
          "Memory size: {} MB {}",
          bytes as f32 / 1024.0 / 1024.0,
          n_mem
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
    }) {
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

  pub fn inference(&mut self, params: LLamaArguments, sender: Sender<InferenceResult>) {
    let num_predict = params
      .num_predict
      .unwrap_or(BigInt::from(512 as u64))
      .get_u64()
      .1 as usize;
    let model = self.model.as_ref().unwrap();
    let vocab = self.vocab.as_ref().unwrap();
    let repeat_last_n = params
      .repeat_last_n
      .unwrap_or(BigInt::from(512 as u64))
      .get_u64()
      .1 as usize;
    let mut session = model.start_session(repeat_last_n);

    let n_threads = params.n_threads.unwrap_or(4);
    let n_batch = params.n_batch.unwrap_or(BigInt::from(8 as u64)).get_u64().1 as usize;
    let top_k = params.top_k.unwrap_or(BigInt::from(30 as u64)).get_u64().1 as usize;

    let inference_params = InferenceParameters {
      n_threads,
      n_batch,
      top_k,
      top_p: params.top_p.unwrap_or(0.95) as f32,
      repeat_penalty: params.repeat_penalty.unwrap_or(1.30) as f32,
      temp: params.temp.unwrap_or(0.8) as f32,
    };

    log::info!("repeat_last_n: {}", repeat_last_n);
    log::info!("n_threads: {}", inference_params.n_threads);
    log::info!("n_batch: {}", inference_params.n_batch);
    log::info!("top_k: {}", inference_params.top_k);
    log::info!("top_p: {}", inference_params.top_p);
    log::info!("repeat_penalty: {}", inference_params.repeat_penalty);
    log::info!("temp: {}", inference_params.temp);

    let seed = if let Some(seed) = params.seed {
      Some(seed.get_u64().1)
    } else {
      None
    };

    log::info!("seed: {:?}", seed);

    let mut rng = if let Some(seed) = seed {
      rand::rngs::StdRng::seed_from_u64(seed)
    } else {
      rand::rngs::StdRng::from_entropy()
    };

    let ended = Arc::new(Mutex::new(false));

    let res = session.inference_with_prompt::<Infallible>(
      &model,
      &vocab,
      &inference_params,
      &params.prompt,
      Some(num_predict),
      &mut rng,
      |t| {
        let sender = sender.clone();
        let ended = ended.clone();
        let to_send = match t {
          OutputToken::Token(token) => InferenceResult::InferenceData(InferenceToken {
            token: token.to_string(),
            completed: false,
          }),
          OutputToken::EndOfText => {
            let mut ended = ended.try_lock().unwrap();
            *ended = true;
            InferenceResult::InferenceData(InferenceToken {
              token: "\n\n<end>\n".to_string(),
              completed: true,
            })
          }
        };

        sender.send(to_send).unwrap();

        Ok(())
      },
    );

    let ended = ended.try_lock().unwrap();

    if *ended == false {
      sender
        .send(InferenceResult::InferenceEnd(Some(
          "Inference terminated".to_string(),
        )))
        .unwrap();
    }

    match res {
      Ok(_) => {
        sender.send(InferenceResult::InferenceEnd(None)).unwrap();
      }
      Err(llama_rs::InferenceError::ContextFull) => {
        sender
          .send(InferenceResult::InferenceEnd(Some(
            "Context window full, stopping inference.".to_string(),
          )))
          .unwrap();
        log::warn!("Context window full, stopping inference.")
      }
      Err(llama_rs::InferenceError::UserCallback(_)) => unreachable!("cannot fail"),
    }
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

  pub fn inference(&self, params: LLamaArguments, sender: Sender<InferenceResult>) {
    self
      .command_sender
      .send(LLamaCommand::Inference(params, sender))
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
            llama.inference(params, sender);
          }
          Ok(LLamaCommand::LoadModel(params, sender)) => {
            llama.load_model(params, sender);
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
