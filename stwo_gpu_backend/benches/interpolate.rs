use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use itertools::Itertools;
use stwo_gpu_backend::cuda::BaseFieldVec;
use stwo_gpu_backend::CudaBackend;
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::poly::circle::{CanonicCoset, CirclePoly, PolyOps};

pub fn gpu_interpolate(c: &mut Criterion) {
    let mut group = c.benchmark_group("interpolate");

    for log_size in 19..=20 {
        let coset = CanonicCoset::new(log_size);
        let values = BaseFieldVec::from_vec((0..coset.size()).map(BaseField::from).collect_vec());
        let evaluations = CudaBackend::new_canonical_ordered(coset, values);
        let twiddle_tree = CudaBackend::precompute_twiddles(coset.half_coset());
        let mut res = CirclePoly::new(BaseFieldVec::from_vec(vec![BaseField::default(), BaseField::default()]));
        group.bench_function(BenchmarkId::new("gpu interpolate", log_size), |b| {
            b.iter(
                || {
                    let new_evaluations = evaluations.clone();
                    res = black_box(CudaBackend::interpolate(new_evaluations, &twiddle_tree));
                },
            )
        });
        assert_eq!(res.coeffs.to_vec().len(), 1 << log_size);
    }
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = gpu_interpolate);
criterion_main!(benches);
