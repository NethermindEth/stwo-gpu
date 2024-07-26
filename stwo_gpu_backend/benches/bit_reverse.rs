use criterion::{criterion_group, criterion_main, Criterion};
use itertools::Itertools;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use stwo_gpu_backend::{cuda::BaseFieldVec, cuda::SecureFieldVec, CudaBackend};
use stwo_prover::core::backend::{Column, ColumnOps};
use stwo_prover::core::fields::{m31::BaseField, qm31::SecureField};

pub fn gpu_bit_reverse_base_field(c: &mut Criterion) {
    const BITS: usize = 28;
    let size = 1 << BITS;
    let mut data = BaseFieldVec::from_vec((0..size).map(BaseField::from).collect_vec());

    c.bench_function(&format!("gpu bit_rev base_field {} bit", BITS), |b| {
        b.iter(|| {
            <CudaBackend as ColumnOps<BaseField>>::bit_reverse_column(&mut data);
        })
    });
}

pub fn gpu_bit_reverse_secure_field(c: &mut Criterion) {
    const BITS: usize = 28;
    let size = 1 << BITS;

    let mut rng = SmallRng::seed_from_u64(0);
    let mut data = SecureFieldVec::from_vec((0..size).map(|_| rng.gen()).collect());
    assert_eq!(data.len(), size);

    c.bench_function(&format!("gpu bit_rev secure_field {} bit", BITS), |b| {
        b.iter(|| {
            <CudaBackend as ColumnOps<SecureField>>::bit_reverse_column(&mut data);
        })
    });
}

criterion_group!(
    name = bit_reverse;
    config = Criterion::default().sample_size(10);
    targets = gpu_bit_reverse_base_field, gpu_bit_reverse_secure_field);
criterion_main!(bit_reverse);
