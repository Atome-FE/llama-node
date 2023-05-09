use std::{
  convert::Infallible,
  fs::File,
  io::{BufReader, BufWriter},
  path::Path,
};

use anyhow::{Error, Result};
use llama_rs::{
  EvaluateOutputRequest, InferenceError, InferenceParameters, InferenceSession,
  InferenceSessionParameters, Model, ModelKVMemoryType, TokenBias, EOT_TOKEN_ID,
};
use rand::SeedableRng;
use zstd::{zstd_safe::CompressionLevel, Decoder, Encoder};

use crate::types::{
  EmbeddingResult, EmbeddingResultType, InferenceResult, InferenceResultType, InferenceToken,
  LLamaConfig, LLamaInferenceArguments, LoadModelResult,
};

const CACHE_COMPRESSION_LEVEL: CompressionLevel = 1;

pub struct LLamaInternal {
  pub model: Option<Model>,
}

fn parse_bias(s: &str) -> Result<TokenBias, String> {
  s.parse()
}

impl LLamaInternal {
  pub async fn load_model(&mut self, params: &LLamaConfig) -> Result<LoadModelResult, napi::Error> {
    let num_ctx_tokens = params.num_ctx_tokens.unwrap_or(512);
    let use_mmap = params.use_mmap.unwrap_or(true);
    log::info!("num_ctx_tokens: {}", num_ctx_tokens);
    // let restore_prompt: Option<String> = None;
    // let cache_prompt: Option<String> = None;
    // let repeat_last_n = 64;
    // let num_predict = Some(128);

    if let Ok(model) = llama_rs::Model::load(
      params.path.clone(),
      use_mmap,
      num_ctx_tokens as usize,
      |progress| {
        use llama_rs::LoadProgress;
        match progress {
          LoadProgress::HyperparametersLoaded(hparams) => {
            log::debug!("Loaded hyperparameters {hparams:#?}")
          }
          LoadProgress::ContextSize { bytes } => log::info!(
            "ggml ctx size = {:.2} MB\n",
            bytes as f64 / (1024.0 * 1024.0)
          ),
          LoadProgress::PartLoading {
            file,
            current_part,
            total_parts,
          } => {
            let current_part = current_part + 1;
            log::info!(
              "Loading model part {}/{} from '{}' (mmap preferred: {})\n",
              current_part,
              total_parts,
              file.to_string_lossy(),
              use_mmap
            )
          }
          LoadProgress::PartTensorLoaded {
            current_tensor,
            tensor_count,
            ..
          } => {
            let current_tensor = current_tensor + 1;
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
      },
    ) {
      self.model = Some(model);

      log::info!("Model fully loaded!");

      Ok(LoadModelResult {
        error: false,
        message: None,
      })
    } else {
      Err(napi::Error::from_reason("Could not load model"))
    }
  }

  pub async fn tokenize(&self, text: &str) -> Result<Vec<i32>, napi::Error> {
    let vocab = self.model.as_ref().unwrap().vocabulary();
    let tokens = vocab
      .tokenize(text, false)
      .unwrap()
      .iter()
      .map(|(_, tid)| *tid)
      .collect::<Vec<_>>();

    Ok(tokens)
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
    };

    log::info!("n_threads: {}", inference_params.n_threads);
    log::info!("n_batch: {}", inference_params.n_batch);
    log::info!("top_k: {}", inference_params.top_k);
    log::info!("top_p: {}", inference_params.top_p);
    log::info!("repeat_penalty: {}", inference_params.repeat_penalty);
    log::info!("temp: {}", inference_params.temperature);

    inference_params
  }

  fn write_session(&self, mut session: InferenceSession, path: &String) -> Result<(), Error> {
    let path = Path::new(path);
    let snap_shot = unsafe { session.get_snapshot() };
    let file = File::create(path)?;
    let encoder = Encoder::new(BufWriter::new(file), CACHE_COMPRESSION_LEVEL)?;
    bincode::serialize_into(encoder.auto_finish(), &snap_shot)?;
    log::info!("Successfully wrote inference session to {path:?}");
    Ok(())
  }

  fn read_or_create_session(
    &self,
    persist_session: Option<&Path>,
    inference_session_params: InferenceSessionParameters,
  ) -> Result<InferenceSession, Error> {
    let model = self.model.as_ref().ok_or(Error::msg("Model not loaded"))?;

    fn load(model: &Model, path: &Path) -> Result<InferenceSession> {
      let file = File::open(path)?;
      let decoder = Decoder::new(BufReader::new(file))?;
      let snapshot = bincode::deserialize_from(decoder)?;
      let session = InferenceSession::from_snapshot(snapshot, model)?;
      log::info!("Loaded inference session from {path:?}");
      Ok(session)
    }

    if let Some(path) = persist_session {
      if path.exists() {
        let session = load(model, path)?;
        Ok(session)
      } else {
        let session = model.start_session(inference_session_params);
        Ok(session)
      }
    } else {
      let session = model.start_session(inference_session_params);
      Ok(session)
    }
  }

  fn start_new_session(&self, params: &LLamaInferenceArguments) -> InferenceSession {
    let repeat_last_n = params.repeat_last_n.unwrap_or(512) as usize;
    let float16 = params.float16.unwrap_or(false);
    let load_session = params.load_session.as_ref().map(Path::new);

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

    // TODO: no unwrap
    self
      .read_or_create_session(load_session, inference_session_params)
      .unwrap()
  }

  pub async fn get_word_embedding(
    &self,
    params: &LLamaInferenceArguments,
  ) -> Result<EmbeddingResult, napi::Error> {
    let mut session = self.start_new_session(params);
    let inference_params = self.get_inference_params(params);
    let model = self.model.as_ref().unwrap();
    let prompt_for_feed = format!(" {}", params.prompt);

    if let Err(InferenceError::ContextFull) =
      session.feed_prompt::<Infallible>(model, &inference_params, prompt_for_feed.as_str(), |_| {
        Ok(())
      })
    {
      return Err(napi::Error::from_reason("Context window full."));
    }

    let end_token = self.tokenize("\n").await.unwrap();

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

    Ok(EmbeddingResult {
      r#type: EmbeddingResultType::Data,
      message: None,
      data: output_request
        .embeddings
        .map(|embd| embd.into_iter().map(|data| data.into()).collect()),
    })
  }

  pub async fn inference(
    &self,
    params: &LLamaInferenceArguments,
    // sender: &Sender<InferenceResult>,
    callback: impl Fn(InferenceResult),
  ) {
    let num_predict = params.num_predict.unwrap_or(512) as usize;
    let model = self.model.as_ref().unwrap();

    let prompt = &params.prompt;
    let feed_prompt_only = params.feed_prompt_only.unwrap_or(false);
    let feed_prompt = if feed_prompt_only {
      true
    } else {
      params.feed_prompt.unwrap_or(false)
    };
    let seed = params.seed.map(|seed| seed as u64);

    let mut session = self.start_new_session(params);
    let inference_params = self.get_inference_params(params);

    let mut rng = if let Some(seed) = seed {
      rand::rngs::StdRng::seed_from_u64(seed)
    } else {
      rand::rngs::StdRng::from_entropy()
    };

    if let Err(InferenceError::ContextFull) =
      session.feed_prompt::<Infallible>(model, &inference_params, prompt, |_| Ok(()))
    {
      callback(InferenceResult {
        r#type: InferenceResultType::Error,
        message: Some("Context window full.".to_string()),
        data: None,
      });
    }

    if !feed_prompt_only {
      let inference_input = if feed_prompt { "" } else { prompt };

      let res = session.inference_with_prompt::<Infallible>(
        model,
        &inference_params,
        inference_input,
        Some(num_predict),
        &mut rng,
        |t| {
          // need stop prompt to handle the model like vicuna
          // https://github.com/rustformers/llama-rs/issues/151
          let to_send = InferenceResult {
            r#type: InferenceResultType::Data,
            message: None,
            data: Some(InferenceToken {
              token: t.to_string(),
              completed: false,
            }),
          };

          callback(to_send);
          
          Ok(())
        },
      );

      match res {
        Ok(_) => {
          let to_send = InferenceResult {
            r#type: InferenceResultType::Data,
            message: None,
            data: Some(InferenceToken {
              token: "\n\n<end>\n".to_string(),
              completed: true,
            }),
          };

          callback(to_send);
        }
        Err(error) => {
          callback(InferenceResult {
            r#type: InferenceResultType::Error,
            message: match error {
              llama_rs::InferenceError::EndOfText => Some("End of text.".to_string()),
              llama_rs::InferenceError::ContextFull => {
                Some("Context window full, stopping inference.".to_string())
              }
              llama_rs::InferenceError::TokenizationFailed => {
                Some("Tokenization failed.".to_string())
              }
              llama_rs::InferenceError::UserCallback(_) => Some("Inference failed.".to_string()),
            },
            data: None,
          });
        }
      }
    } else {
      let to_send = InferenceResult {
        r#type: InferenceResultType::Data,
        message: None,
        data: Some(InferenceToken {
          token: "".to_string(),
          completed: true,
        }),
      };

      callback(to_send);
    }

    if let Some(session_path) = params.save_session.as_ref() {
      self.write_session(session, session_path).unwrap();
    }

    callback(InferenceResult {
      r#type: InferenceResultType::End,
      message: None,
      data: None,
    });
  }
}
