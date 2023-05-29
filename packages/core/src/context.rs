use std::{
    convert::Infallible,
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

use anyhow::{Error, Result};
use llm::{
    InferenceError, InferenceFeedback, InferenceParameters, InferenceSession,
    InferenceSessionConfig, Model, ModelKVMemoryType, OutputRequest, TokenBias,
};
use rand::SeedableRng;
use zstd::{zstd_safe::CompressionLevel, Decoder, Encoder};

use crate::types::{
    Generate, InferenceResult, InferenceResultType, InferenceToken, ModelLoad, ModelType,
};

const CACHE_COMPRESSION_LEVEL: CompressionLevel = 1;

pub struct LLMContext {
    pub model: Box<dyn Model>,
}

// fn parse_bias(s: &str) -> Result<TokenBias, InvalidTokenBias> {
//     s.parse()
// }

impl LLMContext {
    pub async fn load_model(params: &ModelLoad) -> Result<LLMContext, napi::Error> {
        let model = match params.model_type {
            ModelType::Llama => params.load::<llm::models::Llama>(),
            ModelType::Bloom => params.load::<llm::models::Bloom>(),
            ModelType::Gpt2 => params.load::<llm::models::Gpt2>(),
            ModelType::GptJ => params.load::<llm::models::GptJ>(),
            ModelType::GptNeoX => params.load::<llm::models::GptNeoX>(),
            ModelType::Mpt => params.load::<llm::models::Mpt>(),
        }?;

        Ok(LLMContext { model })
    }

    pub async fn tokenize(&self, text: &str) -> Result<Vec<i32>, napi::Error> {
        let vocab = self.model.vocabulary();
        let tokens = vocab
            .tokenize(text, false)
            .map_err(|e| napi::Error::from_reason(format!("Failed to tokenize: {}", e)))?;
        let tokens = tokens.iter().map(|(_, tid)| *tid).collect::<Vec<_>>();

        Ok(tokens)
    }

    fn get_inference_params(&self, params: &Generate) -> InferenceParameters {
        let token_bias = params
            .token_bias
            .as_ref()
            .map(|pairs| {
                let pairs = pairs
                    .iter()
                    .map(|pair| (pair.token_id, pair.bias as f32))
                    .collect::<Vec<_>>();
                TokenBias::new(pairs)
            })
            .unwrap_or_else(|| {
                if params.ignore_eos {
                    TokenBias::new(vec![(self.model.eot_token_id(), -1.0)])
                } else {
                    TokenBias::default()
                }
            });

        let inference_params = InferenceParameters {
            n_threads: params.num_threads as usize,
            n_batch: params.batch_size as usize,
            top_k: params.top_k as usize,
            top_p: params.top_p as f32,
            repeat_penalty: params.repeat_penalty as f32,
            temperature: params.temperature as f32,
            bias_tokens: token_bias,
            repetition_penalty_last_n: params.repeat_last_n as usize,
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
        inference_session_config: InferenceSessionConfig,
    ) -> Result<InferenceSession, Error> {
        let model = self.model.as_ref();

        fn load(model: &dyn Model, path: &Path) -> Result<InferenceSession> {
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
                let session = model.start_session(inference_session_config);
                Ok(session)
            }
        } else {
            let session = model.start_session(inference_session_config);
            Ok(session)
        }
    }

    fn start_session(&self, params: &Generate) -> Result<InferenceSession> {
        let float16 = params.float16;
        let load_session = params.load_session.as_ref().map(Path::new);

        let inference_session_params = {
            let mem_typ = if float16 {
                ModelKVMemoryType::Float16
            } else {
                ModelKVMemoryType::Float32
            };
            InferenceSessionConfig {
                memory_k_type: mem_typ,
                memory_v_type: mem_typ,
            }
        };

        self.read_or_create_session(load_session, inference_session_params)
    }

    pub async fn get_word_embedding(&self, params: &Generate) -> Result<Vec<f64>, napi::Error> {
        let mut session = self.start_session(params).map_err(|e| {
            napi::Error::from_reason(format!("Failed to start inference session: {}", e))
        })?;
        let inference_params = self.get_inference_params(params);
        let model = self.model.as_ref();
        let prompt_for_feed = format!(" {}", params.prompt);

        if let Err(InferenceError::ContextFull) = session.feed_prompt::<Infallible, &str>(
            model,
            &inference_params,
            prompt_for_feed.as_str(),
            &mut Default::default(),
            |_| Ok(InferenceFeedback::Continue),
        ) {
            return Err(napi::Error::from_reason("Context window full."));
        }

        let end_token = self.tokenize("\n").await.map_err(|e| {
            napi::Error::from_reason(format!("Failed to tokenize end token: {}", e))
        })?;

        let mut output_request = OutputRequest {
            all_logits: None,
            embeddings: Some(Vec::new()),
        };

        model.evaluate(
            &mut session,
            &inference_params,
            &end_token,
            &mut output_request,
        );

        let output: Option<Vec<f64>> = output_request
            .embeddings
            .map(|embd| embd.into_iter().map(|data| data.into()).collect());

        Ok(output.unwrap_or(Vec::new()))
    }

    pub fn inference(
        &self,
        params: &Generate,
        callback: impl Fn(InferenceResult) -> InferenceFeedback,
    ) -> Result<(), napi::Error> {
        let prompt = &params.prompt;
        let inference_params = self.get_inference_params(params);
        let model = self.model.as_ref();

        let feed_prompt_only = params.feed_prompt_only;

        let feed_prompt = if feed_prompt_only {
            true
        } else {
            params.feed_prompt
        };

        let seed = params.seed.map(|seed| seed as u64);

        let mut session = self.start_session(params).map_err(|e| {
            napi::Error::from_reason(format!("Failed to start inference session: {}", e))
        })?;

        let maximum_token_count = if feed_prompt_only {
            Some(0)
        } else {
            Some(params.num_predict as usize)
        };

        let mut rng = if let Some(seed) = seed {
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::rngs::StdRng::from_entropy()
        };

        let res = session.infer::<Infallible>(
            model,
            &mut rng,
            &llm::InferenceRequest {
                prompt: llm::Prompt::Text(prompt),
                parameters: &inference_params,
                play_back_previous_tokens: !feed_prompt,
                maximum_token_count,
            },
            &mut Default::default(),
            |r| match &r {
                llm::InferenceResponse::PromptToken(t) => {
                    let to_send = InferenceResult {
                        r#type: InferenceResultType::Data,
                        message: None,
                        data: Some(InferenceToken {
                            token: t.to_string(),
                            completed: false,
                        }),
                    };

                    Ok(if feed_prompt {
                        InferenceFeedback::Continue
                    } else {
                        callback(to_send)
                    })
                }
                llm::InferenceResponse::InferredToken(t) => {
                    let to_send = InferenceResult {
                        r#type: InferenceResultType::Data,
                        message: None,
                        data: Some(InferenceToken {
                            token: t.to_string(),
                            completed: false,
                        }),
                    };

                    Ok(if feed_prompt_only {
                        InferenceFeedback::Continue
                    } else {
                        callback(to_send)
                    })
                }
                llm::InferenceResponse::SnapshotToken(_) => Ok(InferenceFeedback::Continue),
                llm::InferenceResponse::EotToken => Ok(InferenceFeedback::Continue),
            },
        );

        match res {
            Ok(_state) => {
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
                let message = match error {
                    InferenceError::EndOfText => "End of text.".to_string(),
                    InferenceError::ContextFull => {
                        "Context window full, stopping inference.".to_string()
                    }
                    InferenceError::TokenizationFailed(_) => "Tokenization failed.".to_string(),
                    InferenceError::UserCallback(_) => "Inference failed.".to_string(),
                };
                callback(InferenceResult {
                    r#type: InferenceResultType::Error,
                    message: Some(message),
                    data: None,
                });
            }
        }

        if let Some(session_path) = params.save_session.as_ref() {
            self.write_session(session, session_path).map_err(|e| {
                napi::Error::from_reason(format!("Failed to write inference session: {}", e))
            })?;
        }

        callback(InferenceResult {
            r#type: InferenceResultType::End,
            message: None,
            data: None,
        });

        Ok(())
    }
}
