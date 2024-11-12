use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use itertools::Itertools;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use stwo_gpu_backend::cuda::{BaseFieldVec, SecureFieldVec};
use stwo_gpu_backend::CudaBackend;
use stwo_prover::core::backend::simd::column::{BaseColumn, SecureColumn};
use stwo_prover::core::backend::simd::SimdBackend;
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::fields::qm31::SecureField;
use stwo_prover::core::fields::FieldOps;

pub fn gpu_batch_inverse(c: &mut Criterion) {
    for bits in 15..29 {
        let size = 1 << bits;

        let mut rng = SmallRng::seed_from_u64(0);
        let data = BaseFieldVec::from_vec((0..size).map(|_| rng.gen()).collect_vec());
        let mut res = data.clone();
        c.bench_function(
            &format!("gpu batch_inverse base field {} bits", bits),
            |b| {
                b.iter(|| <CudaBackend as FieldOps<BaseField>>::batch_inverse(&data, &mut res));
            },
        );
    }
}

pub fn simd_batch_inverse(c: &mut Criterion) {
    for bits in 15..29 {
        let size = 1 << bits;
        let data = (0..size).map(BaseField::from).collect::<BaseColumn>();
        let mut res = data.clone();
        c.bench_function(
            &format!("simd batch_inverse base field {} bits", bits),
            |b| {
                b.iter(|| <SimdBackend as FieldOps<BaseField>>::batch_inverse(&data, &mut res));
            },
        );
    }
}

pub fn gpu_batch_inverse_secure_field(c: &mut Criterion) {
    for bits in 15..29 {
        let size = 1 << bits;

        let mut rng = SmallRng::seed_from_u64(0);
        let data = SecureFieldVec::from_vec((0..size).map(|_| rng.gen()).collect());

        let mut res = data.clone();
        c.bench_function(
            &format!("gpu batch_inverse secure field {} bits", bits),
            |b| {
                b.iter(|| <CudaBackend as FieldOps<SecureField>>::batch_inverse(&data, &mut res));
            },
        );
    }
}

pub fn simd_batch_inverse_secure_field(c: &mut Criterion) {
    for bits in 15..29 {
        let size = 1 << bits;

        let mut rng = SmallRng::seed_from_u64(0);
        let data: SecureColumn = (0..size)
        .map(|_| rng.gen())
        .collect::<Vec<SecureField>>()
        .into_iter()
        .collect();

        let mut res = data.clone();
        c.bench_function(
            &format!("simd batch_inverse secure field {} bits", bits),
            |b| {
                b.iter(|| <SimdBackend as FieldOps<SecureField>>::batch_inverse(&data, &mut res));
            },
        );
    }
}

criterion_group!(
    name = batch_inverse;
    config = Criterion::default().sample_size(10);
    targets = gpu_batch_inverse, simd_batch_inverse, gpu_batch_inverse_secure_field, simd_batch_inverse_secure_field);
criterion_main!(batch_inverse);
