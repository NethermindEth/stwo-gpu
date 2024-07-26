use criterion::{criterion_group, criterion_main, Criterion};
use itertools::Itertools;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use stwo_gpu_backend::cuda::{BaseFieldVec, SecureFieldVec};
use stwo_gpu_backend::CudaBackend;
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::fields::qm31::SecureField;
use stwo_prover::core::fields::FieldOps;

pub fn gpu_batch_inverse(c: &mut Criterion) {
    const BITS: usize = 28;
    const SIZE: usize = 1 << BITS;

    let data = BaseFieldVec::from_vec((0..SIZE).map(BaseField::from).collect_vec());
    let mut res = data.clone();
    c.bench_function(
        &format!("gpu batch_inverse base field {} bits", BITS),
        |b| {
            b.iter(|| <CudaBackend as FieldOps<BaseField>>::batch_inverse(&data, &mut res));
        },
    );
}

pub fn gpu_batch_inverse_secure_field(c: &mut Criterion) {
    const BITS: usize = 28;
    let size = 1 << BITS;

    let mut rng = SmallRng::seed_from_u64(0);
    let data = SecureFieldVec::from_vec((0..size).map(|_| rng.gen()).collect());

    let mut res = data.clone();
    c.bench_function(
        &format!("gpu batch_inverse secure field {} bits", BITS),
        |b| {
            b.iter(|| <CudaBackend as FieldOps<SecureField>>::batch_inverse(&data, &mut res));
        },
    );
}

criterion_group!(
    name = batch_inverse;
    config = Criterion::default().sample_size(10);
    targets = gpu_batch_inverse, gpu_batch_inverse_secure_field);
criterion_main!(batch_inverse);
