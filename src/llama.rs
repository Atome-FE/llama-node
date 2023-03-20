use std::{
  convert::Infallible,
  sync::{
    mpsc::{channel, Receiver, Sender, TryRecvError},
    Arc, Mutex,
  },
  thread,
};

use llama_rs::{InferenceParameters, Model, OutputToken, Vocabulary};
use napi::bindgen_prelude::BigInt;
use rand::SeedableRng;

#[napi(object)]
#[derive(Clone)]
pub struct InferenceToken {
  pub token: String,
  pub completed: bool,
}

#[derive(Clone)]
pub enum InferenceCallback {
  InferenceData(InferenceToken),
  InferenceError(String),
  Terminate,
}

#[napi(object)]
#[derive(Clone)]
pub struct LLamaConfig {
  pub path: String,
  pub num_ctx_tokens: Option<i32>,
}

#[napi(object)]
#[derive(Clone)]
pub struct LLamaArguments {
  pub n_threads: Option<i32>,
  pub n_batch: Option<BigInt>,
  pub top_k: Option<BigInt>,
  pub top_p: Option<f64>,
  pub repeat_penalty: Option<f64>,
  pub temp: Option<f64>,
  pub seed: Option<BigInt>,
  pub num_predict: Option<BigInt>,
  pub repeat_last_n: Option<BigInt>,
  pub prompt: String,
}

#[derive(Clone)]
enum LLamaCommand {
  LoadModel(LLamaConfig),
  Inference(LLamaArguments),
  Terminate,
}

pub struct LLamaChannel {
  main_thread_sender: Sender<InferenceCallback>,
  llama_sender: Sender<LLamaCommand>,
  llama_receiver: Arc<Mutex<Receiver<LLamaCommand>>>,
}

impl LLamaChannel {
  pub fn new(main_thread_sender: Sender<InferenceCallback>) -> Self {
    let (llama_sender, llama_receiver) = channel::<LLamaCommand>();

    let channel = LLamaChannel {
      llama_receiver: Arc::new(Mutex::new(llama_receiver)),
      llama_sender,
      main_thread_sender,
    };

    channel.spawn();

    channel
  }

  pub fn load_model(&self, params: LLamaConfig) {
    self
      .llama_sender
      .send(LLamaCommand::LoadModel(params))
      .unwrap();
  }

  pub fn inference(&self, params: LLamaArguments) {
    self
      .llama_sender
      .send(LLamaCommand::Inference(params))
      .unwrap();
  }

  pub fn terminate(&self) {
    self.llama_sender.send(LLamaCommand::Terminate).unwrap();
  }

  pub fn spawn(&self) {
    let rv = self.llama_receiver.clone();
    let tx = self.main_thread_sender.clone();

    thread::spawn(move || {
      struct LLamaInternal {
        model: Option<Model>,
        vocab: Option<Vocabulary>,
        sender: Sender<InferenceCallback>,
      }

      impl LLamaInternal {
        pub fn load_model(&mut self, params: LLamaConfig) {
          env_logger::builder()
            .filter_level(log::LevelFilter::Info)
            .parse_default_env()
            .init();

          let num_ctx_tokens = params.num_ctx_tokens.unwrap_or(512);
          // let restore_prompt: Option<String> = None;
          // let cache_prompt: Option<String> = None;
          // let repeat_last_n = 64;
          // let num_predict = Some(128);

          let (model, vocab) = llama_rs::Model::load(params.path, num_ctx_tokens, |progress| {
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
          })
          .expect("Could not load model");

          self.model = Some(model);
          self.vocab = Some(vocab);

          log::info!("Model fully loaded!");
        }

        pub fn inference(&mut self, params: LLamaArguments) {
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

          let seed: Option<u64> = Some(params.seed.unwrap_or(BigInt::from(0 as u64)).get_u64().1);

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
              self
                .sender
                .send(match t {
                  OutputToken::Token(token) => InferenceCallback::InferenceData(InferenceToken {
                    token: token.to_string(),
                    completed: false,
                  }),
                  OutputToken::EndOfText => {
                    let mut ended = ended.try_lock().unwrap();
                    *ended = true;
                    InferenceCallback::InferenceData(InferenceToken {
                      token: "\n\n<end>\n".to_string(),
                      completed: true,
                    })
                  }
                })
                .unwrap();

              Ok(())
            },
          );

          let ended = ended.try_lock().unwrap();
          if *ended == false {
            self
              .sender
              .send(InferenceCallback::InferenceError(
                "Inference terminated".to_string(),
              ))
              .unwrap()
          }

          match res {
            Ok(_) => (),
            Err(llama_rs::InferenceError::ContextFull) => {
              self
                .sender
                .send(InferenceCallback::InferenceError(
                  "Context window full, stopping inference.".to_string(),
                ))
                .unwrap();
              log::warn!("Context window full, stopping inference.")
            }
            Err(llama_rs::InferenceError::UserCallback(_)) => unreachable!("cannot fail"),
          }
        }
      }

      let mut llama = LLamaInternal {
        model: None,
        vocab: None,
        sender: tx,
      };

      let rv = rv.lock().unwrap();

      loop {
        let command = rv.try_recv();
        match command {
          Ok(LLamaCommand::Inference(params)) => llama.inference(params),
          Ok(LLamaCommand::LoadModel(params)) => llama.load_model(params),
          Err(TryRecvError::Disconnected) | Ok(LLamaCommand::Terminate) => {
            break;
          }
          Err(_) => {
            thread::yield_now();
          }
        }
      }
    });
  }
}
