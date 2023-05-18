use log::{LevelFilter, Log, Metadata, Record};
use once_cell::sync::Lazy;

pub struct LLamaLogger {
    enabled: bool,
}

static mut LLAMA_LOGGER_INNER: LLamaLogger = LLamaLogger { enabled: true };
pub static mut LLAMA_LOGGER: Lazy<&mut LLamaLogger> = Lazy::new(|| {
    log::set_max_level(LevelFilter::Info);
    log::set_logger(unsafe { &LLAMA_LOGGER_INNER }).unwrap();
    unsafe { &mut LLAMA_LOGGER_INNER }
});

impl LLamaLogger {
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn get_singleton() -> &'static mut LLamaLogger {
        unsafe { &mut LLAMA_LOGGER }
    }
}

impl Log for LLamaLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= log::Level::Info
        // true
    }

    fn log(&self, record: &Record) {
        // Check if the record is matched by the logger before logging
        if self.enabled(record.metadata()) && self.enabled {
            println!("{} - {}", record.level(), record.args());
        }
    }

    fn flush(&self) {}
}
