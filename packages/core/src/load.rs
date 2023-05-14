use std::path::Path;

use crate::types::ModelLoad;
use anyhow::Result;
use llm::{load, LoadProgress, Model, ModelParameters};

impl ModelLoad {
    pub fn load<M: llm::KnownModel + 'static>(&self) -> Result<Box<dyn Model>, napi::Error> {
        let params = ModelParameters {
            prefer_mmap: self.use_mmap.unwrap_or(true),
            n_context_tokens: self.num_ctx_tokens.unwrap_or(2048) as usize,
            ..Default::default()
        };

        let path = Path::new(&self.model_path);

        let now = std::time::Instant::now();
        let mut prev_load_time = now;

        let model = load::<M>(path, params, None, move |progress| match progress {
            LoadProgress::HyperparametersLoaded => {
                log::info!("Loaded hyperparameters")
            }
            LoadProgress::ContextSize { bytes } => log::info!(
                "ggml ctx size = {:.2} MB\n",
                bytesize::to_string(bytes as u64, false)
            ),
            LoadProgress::TensorLoaded {
                current_tensor,
                tensor_count,
                ..
            } => {
                if prev_load_time.elapsed().as_millis() > 500 {
                    // We don't want to re-render this on every message, as that causes the
                    // spinner to constantly reset and not look like it's spinning (and
                    // it's obviously wasteful).

                    log::info!(
                        "{}",
                        format!("Loaded tensor {}/{}", current_tensor + 1, tensor_count)
                    );

                    prev_load_time = std::time::Instant::now();
                }
            }
            LoadProgress::Loaded {
                file_size,
                tensor_count,
            } => {
                log::info!(
                    "{}",
                    &format!(
                        "Loaded {tensor_count} tensors ({}) after {}ms",
                        bytesize::to_string(file_size, false),
                        now.elapsed().as_millis()
                    )
                );
            }
            LoadProgress::LoraApplied { name } => {
                log::info!("Applied Lora: {}", name);
            }
        })
        .map_err(|e| napi::Error::from_reason(format!("{}", e)))?;

        Ok(Box::new(model))
    }
}
