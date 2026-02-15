pub mod attention;
pub mod buffer;
pub mod context;
pub mod gemm;

pub use attention::{attention_fp16, AttnParams};
pub use buffer::MetalBuffer;
pub use context::MetalContext;

pub type Result<T> = std::result::Result<T, MetalError>;

#[derive(thiserror::Error, Debug)]
pub enum MetalError {
    #[error("metal error: {0}")]
    Metal(&'static str),
    #[error("invalid arg: {0}")]
    Invalid(&'static str),
    #[error("compile error: {0}")]
    Compile(String),
}
