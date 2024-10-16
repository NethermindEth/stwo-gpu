use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use itertools::Itertools;
use stwo_gpu_backend::cuda::BaseFieldVec;
use stwo_gpu_backend::CudaBackend;
use stwo_prover::core::backend::simd::SimdBackend;
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::poly::circle::{CanonicCoset, CircleEvaluation, PolyOps};
use stwo_prover::core::poly::BitReversedOrder;
use stwo_prover::core::ColumnVec;

const LOG_COLUMN_SIZE: u32 = 10;
const LOG_NUMBER_OF_COLUMNS: usize = 16;
const LOG_BLOWUP_FACTOR: u32 = 2;

pub fn simd_evaluate_columns(c: &mut Criterion) {
    let mut group = c.benchmark_group("evaluate_columns");

    let coset = CanonicCoset::new(LOG_COLUMN_SIZE);
    let values = (0..coset.size()).map(BaseField::from).collect();
    let circle_evaluation: CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> =
        SimdBackend::new_canonical_ordered(coset, values);

    let interpolation_coset = CanonicCoset::new(LOG_COLUMN_SIZE + LOG_BLOWUP_FACTOR);
    let twiddle_tree = SimdBackend::precompute_twiddles(interpolation_coset.half_coset());

    let polynomial = SimdBackend::interpolate(circle_evaluation, &twiddle_tree);
    let polynomials = ColumnVec::from(vec![polynomial; 1 << LOG_NUMBER_OF_COLUMNS]);

    group.bench_function(BenchmarkId::new("simd evaluate", LOG_COLUMN_SIZE), |b| {
        b.iter_batched(
            || polynomials.clone(),
            |mut polynomial| {
                SimdBackend::evaluate_polynomials(&mut polynomial, LOG_BLOWUP_FACTOR, &twiddle_tree)
            },
            BatchSize::LargeInput,
        )
    });
}

pub fn gpu_evaluate_columns(c: &mut Criterion) {
    let mut group = c.benchmark_group("evaluate_columns");

    let coset = CanonicCoset::new(LOG_COLUMN_SIZE);
    let values = BaseFieldVec::from_vec((0..coset.size()).map(BaseField::from).collect_vec());
    let circle_evaluation: CircleEvaluation<CudaBackend, BaseField, BitReversedOrder> =
        CudaBackend::new_canonical_ordered(coset, values);

    let interpolation_coset = CanonicCoset::new(LOG_COLUMN_SIZE + LOG_BLOWUP_FACTOR);
    let twiddle_tree = CudaBackend::precompute_twiddles(interpolation_coset.half_coset());

    let polynomial = CudaBackend::interpolate(circle_evaluation, &twiddle_tree);
    let polynomials = ColumnVec::from(vec![polynomial; 1 << LOG_NUMBER_OF_COLUMNS]);

    group.bench_function(BenchmarkId::new("gpu evaluate", LOG_COLUMN_SIZE), |b| {
        b.iter_batched(
            || polynomials.clone(),
            |mut polynomial| {
                CudaBackend::evaluate_polynomials(&mut polynomial, LOG_BLOWUP_FACTOR, &twiddle_tree)
            },
            BatchSize::LargeInput,
        )
    });
}

criterion_group!(
    name = interpolate_columns;
    config = Criterion::default().sample_size(10);
    targets = simd_evaluate_columns, gpu_evaluate_columns);
criterion_main!(interpolate_columns);
