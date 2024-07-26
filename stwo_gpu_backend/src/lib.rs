mod accumulation;
pub mod backend;
mod column;
pub mod cuda;
mod field;
mod fri;
mod poly;
mod quotient;

use cuda::BaseFieldVec;
use cuda::SecureFieldVec;
pub use backend::CudaBackend;
