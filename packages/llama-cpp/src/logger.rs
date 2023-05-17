use log::{LevelFilter, Log, Metadata, Record};
use once_cell::sync::Lazy;

pub struct LLamaLogger {
    enabled: bool,
}

pub static mut LLAMA_LOGGER: LLamaLogger = LLamaLogger { enabled: true };
pub static LLAMA_LOGGER_LOADED: Lazy<bool> = Lazy::new(|| {
    log::set_max_level(LevelFilter::Info);
    log::set_logger(unsafe { &LLAMA_LOGGER }).unwrap();
    true
});

impl LLamaLogger {
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

impl Log for LLamaLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= log::Level::Info
    }

    fn log(&self, record: &Record) {
        // Check if the record is matched by the logger before logging
        if self.enabled(record.metadata()) && self.enabled {
            println!("{} - {}", record.level(), record.args());
        }
    }

    fn flush(&self) {}
}
