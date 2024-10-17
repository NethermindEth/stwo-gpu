use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use itertools::Itertools;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use stwo_gpu_backend::cuda::BaseFieldVec;
use stwo_gpu_backend::CudaBackend;
use stwo_prover::core::air::mask::fixed_mask_points;
use stwo_prover::core::backend::simd::SimdBackend;
use stwo_prover::core::circle::CirclePoint;
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::fields::qm31::SecureField;
use stwo_prover::core::pcs::TreeVec;
use stwo_prover::core::poly::circle::{CanonicCoset, CircleEvaluation, PolyOps};
use stwo_prover::core::poly::BitReversedOrder;
use stwo_prover::core::ColumnVec;

const LOG_COLUMN_SIZE: u32 = 16;
const LOG_NUMBER_OF_COLUMNS: usize = 10;
const LOG_BLOWUP_FACTOR: u32 = 2;

fn generate_random_point() -> CirclePoint<SecureField> {
    let mut rng = SmallRng::seed_from_u64(0);
    let x = rng.gen();
    let y = rng.gen();
    CirclePoint { x, y }
}

fn mask_points(
    point: CirclePoint<SecureField>,
    number_of_columns: usize,
) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
    TreeVec(vec![fixed_mask_points(
        &vec![vec![0_usize]; number_of_columns],
        point,
    )])
}

pub fn simd_evaluate_polynomials_out_of_domain(c: &mut Criterion) {
    let mut group = c.benchmark_group("evaluate_polynomials_out_of_domain");

    let number_of_columns =  1 << LOG_NUMBER_OF_COLUMNS;

    let coset = CanonicCoset::new(LOG_COLUMN_SIZE);
    let values = (0..coset.size()).map(BaseField::from).collect();
    let circle_evaluation: CircleEvaluation<SimdBackend, BaseField, BitReversedOrder> =
        SimdBackend::new_canonical_ordered(coset, values);

    let interpolation_coset = CanonicCoset::new(LOG_COLUMN_SIZE + LOG_BLOWUP_FACTOR);
    let twiddle_tree = SimdBackend::precompute_twiddles(interpolation_coset.half_coset());

    let polynomial = SimdBackend::interpolate(circle_evaluation, &twiddle_tree);
    let polynomials = ColumnVec::from(vec![&polynomial; number_of_columns]);

    let point = generate_random_point();
    let sample_points = mask_points(point, number_of_columns);

    group.bench_function(BenchmarkId::new("gpu evaluate polynomials out of domain", LOG_COLUMN_SIZE), |b| {
        b.iter_batched(
            || (polynomials.clone(), sample_points.clone()),
            |(polys, sample)| {
                SimdBackend::evaluate_polynomials_out_of_domain(TreeVec::new(vec![polys]), sample)
            },
            BatchSize::LargeInput,
        )
    });
}

pub fn gpu_evaluate_polynomials_out_of_domain(c: &mut Criterion) {
    let mut group = c.benchmark_group("evaluate_polynomials_out_of_domain");

    let number_of_columns =  1 << LOG_NUMBER_OF_COLUMNS;

    let coset = CanonicCoset::new(LOG_COLUMN_SIZE);
    let values = BaseFieldVec::from_vec((0..coset.size()).map(BaseField::from).collect_vec());
    let circle_evaluation: CircleEvaluation<CudaBackend, BaseField, BitReversedOrder> =
        CudaBackend::new_canonical_ordered(coset, values);

    let interpolation_coset = CanonicCoset::new(LOG_COLUMN_SIZE + LOG_BLOWUP_FACTOR);
    let twiddle_tree = CudaBackend::precompute_twiddles(interpolation_coset.half_coset());

    let polynomial = CudaBackend::interpolate(circle_evaluation, &twiddle_tree);
    let polynomials = ColumnVec::from(vec![&polynomial; number_of_columns]);

    let point = generate_random_point();
    let sample_points = mask_points(point, number_of_columns);

    group.bench_function(BenchmarkId::new("simd evaluate polynomials out of domain", LOG_COLUMN_SIZE), |b| {
        b.iter_batched(
            || (polynomials.clone(), sample_points.clone()),
            |(polys, sample)| {
                CudaBackend::evaluate_polynomials_out_of_domain(TreeVec::new(vec![polys]), sample)
            },
            BatchSize::LargeInput,
        )
    });
}

criterion_group!(
    name = evaluate_polynomials_out_of_domain;
    config = Criterion::default().sample_size(10);
    targets = simd_evaluate_polynomials_out_of_domain, gpu_evaluate_polynomials_out_of_domain
);
criterion_main!(evaluate_polynomials_out_of_domain);
