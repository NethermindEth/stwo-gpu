#[cfg(test)]
mod tests {
    use crate::CudaBackend;
    use stwo_prover::core::backend::simd::SimdBackend;
    use stwo_prover::core::fields::m31::BaseField;
    use stwo_prover::core::fri::FriConfig;
    use stwo_prover::core::pcs::{CommitmentSchemeProver, PcsConfig};
    use stwo_prover::core::poly::circle::{CanonicCoset, CircleEvaluation, PolyOps};
    use stwo_prover::core::poly::BitReversedOrder;
    use stwo_prover::core::vcs::blake2_merkle::Blake2sMerkleChannel;
    use stwo_prover::core::ColumnVec;
    use tracing::{span, Level};

    const LOG_EXPAND: u32 = 2;

    #[test_log::test]
    fn test_interpolation_for_wide_trace() {
        let number_of_columns: usize = 1 << 15;
        let column_size: u32 = 10;

        let blowup_factor = 2;

        measure_interpolation_time_gpu_vs_simd(blowup_factor, number_of_columns, column_size);
    }

    #[test_log::test]
    fn test_interpolation_for_narrow_trace() {
        let number_of_columns: usize = 1 << 5;
        let column_size: u32 = 20;

        let blowup_factor = 2;

        measure_interpolation_time_gpu_vs_simd(blowup_factor, number_of_columns, column_size)
    }

    fn measure_interpolation_time_gpu_vs_simd(
        blowup_factor: u32, number_of_columns: usize, column_size: u32,
    ) {
        let domain = CanonicCoset::new(
            column_size + LOG_EXPAND + blowup_factor
        ).circle_domain();
        let config = PcsConfig {
            pow_bits: 10,
            fri_config: FriConfig::new(5, 1, 64),
        };

        let simd_columns: ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> =
            (0..number_of_columns).map(|_index|
                CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(
                    domain, (0..domain.size()).map(BaseField::from).collect(),
                )
            ).collect();
        let simd_twiddles = SimdBackend::precompute_twiddles(domain.half_coset);
        let mut simd_commitment_scheme_prover: CommitmentSchemeProver<'_, SimdBackend, Blake2sMerkleChannel> =
            CommitmentSchemeProver::new(config, &simd_twiddles);
        let simd_span = span!(Level::INFO, "Test SIMD interpolation").entered();
        simd_commitment_scheme_prover.tree_builder().extend_evals(simd_columns);
        simd_span.exit();

        let gpu_columns: ColumnVec<CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>> =
            (0..number_of_columns).map(|_index|
                CircleEvaluation::<CudaBackend, BaseField, BitReversedOrder>::new(
                    domain, (0..domain.size()).map(BaseField::from).collect(),
                )
            ).collect();
        let gpu_twiddles = CudaBackend::precompute_twiddles(domain.half_coset);
        let mut gpu_commitment_scheme_prover: CommitmentSchemeProver<'_, CudaBackend, Blake2sMerkleChannel> =
            CommitmentSchemeProver::new(config, &gpu_twiddles);
        let gpu_span = span!(Level::INFO, "Test GPU interpolation").entered();
        gpu_commitment_scheme_prover.tree_builder().extend_evals(gpu_columns);
        gpu_span.exit();
    }
}