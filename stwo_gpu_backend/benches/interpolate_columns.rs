use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use itertools::Itertools;
use stwo_gpu_backend::cuda::BaseFieldVec;
use stwo_gpu_backend::CudaBackend;
use stwo_prover::core::backend::simd::SimdBackend;
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::poly::circle::{CanonicCoset, CircleEvaluation, PolyOps};
use stwo_prover::core::poly::BitReversedOrder;

const LOG_COLUMN_SIZE: u32 = 10;
const LOG_NUMBER_OF_COLUMNS: usize = 16;

pub fn simd_interpolate_columns(c: &mut Criterion) {
    let mut group = c.benchmark_group("interpolate_columns");

    let coset = CanonicCoset::new(LOG_COLUMN_SIZE);
    let values = (0..coset.size()).map(BaseField::from).collect();
    let twiddle_tree = SimdBackend::precompute_twiddles(coset.half_coset());
    let circle_evaluation: CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> =
        SimdBackend::new_canonical_ordered(coset, values);
    let evaluations = vec![circle_evaluation; 1 << LOG_NUMBER_OF_COLUMNS];

    group.bench_function(BenchmarkId::new("simd interpolate", LOG_COLUMN_SIZE), |b| {
        b.iter_batched(
            || evaluations.clone(),
            |evaluations| SimdBackend::interpolate_columns(evaluations, &twiddle_tree),
            BatchSize::LargeInput,
        )
    });
}

pub fn gpu_interpolate_columns(c: &mut Criterion) {
    let mut group = c.benchmark_group("interpolate_columns");

    let coset = CanonicCoset::new(LOG_COLUMN_SIZE);
    let values = BaseFieldVec::from_vec((0..coset.size()).map(BaseField::from).collect_vec());
    let twiddle_tree = CudaBackend::precompute_twiddles(coset.half_coset());
    let circle_evaluation: CircleEvaluation<CudaBackend, BaseField, BitReversedOrder> =
        CudaBackend::new_canonical_ordered(coset, values);
    let evaluations = vec![circle_evaluation; 1 << LOG_NUMBER_OF_COLUMNS];

    group.bench_function(BenchmarkId::new("gpu interpolate", LOG_COLUMN_SIZE), |b| {
        b.iter_batched(
            || evaluations.clone(),
            |evaluations| CudaBackend::interpolate_columns(evaluations, &twiddle_tree),
            BatchSize::LargeInput,
        )
    });
}

criterion_group!(
    name = interpolate_columns;
    config = Criterion::default().sample_size(10);
    targets = simd_interpolate_columns, gpu_interpolate_columns);
criterion_main!(interpolate_columns);
