mod accumulation;
pub mod backend;
mod column;
pub mod cuda;
mod field;
mod fri;
mod poly;
mod quotient;

pub use backend::CudaBackend;


#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rand::{rngs::SmallRng, SeedableRng};
    use stwo_prover::core::{fields::m31::BaseField, poly::circle::{CanonicCoset, PolyOps}};
    use tracing::{span, Level};

    use crate::{cuda::{self}, CudaBackend};
    
    #[test_log::test]
    fn test_with_log() {
        let log_size = 28;

        let size = 1 << log_size;

        let data = cuda::BaseFieldVec::from_vec((0..size).map(BaseField::from).collect_vec());

        let coset = CanonicCoset::new(log_size);
        let span = span!(Level::INFO, "new_canonical_ordered").entered();
        let gpu_evaluations = CudaBackend::new_canonical_ordered(coset, data);
        span.exit();

        let span = span!(Level::INFO, "precompute_twiddles").entered();
        let twiddles = CudaBackend::precompute_twiddles(coset.half_coset());
        span.exit();

        let span = span!(Level::INFO, "interpolate").entered();
        let _ = CudaBackend::interpolate(gpu_evaluations, &twiddles);
        span.exit();
    }
}