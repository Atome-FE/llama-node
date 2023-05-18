use console::style;
use log::{Level, LevelFilter, Log, Metadata, Record};
use once_cell::sync::Lazy;
use time::{format_description, OffsetDateTime};

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

    pub fn style_level(&self, level: Level) -> console::StyledObject<&str> {
        match level {
            Level::Error => style("ERROR").red(),
            Level::Warn => style("WARN").yellow(),
            Level::Info => style("INFO").green(),
            Level::Debug => style("DEBUG").blue(),
            Level::Trace => style("TRACE").cyan(),
        }
    }
}

impl Log for LLamaLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= log::Level::Info
    }

    fn log(&self, record: &Record) {
        // Check if the record is matched by the logger before logging
        if self.enabled(record.metadata()) && self.enabled {
            let time = OffsetDateTime::now_utc()
                .format(&format_description::well_known::Rfc2822)
                .unwrap();

            println!(
                "[{} - {} - {}] - {}",
                time,
                self.style_level(record.level()),
                record.module_path().unwrap_or("unknown"),
                record.args()
            );
        }
    }

    fn flush(&self) {}
}
