mod accumulation;
pub mod backend;
mod column;
pub mod cuda;
mod field;
mod fri;
mod poly;
mod quotient;

pub use backend::CudaBackend;
use cuda::BaseFieldVec;
use cuda::SecureFieldVec;
