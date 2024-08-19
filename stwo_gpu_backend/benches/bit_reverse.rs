use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use itertools::Itertools;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use stwo_gpu_backend::{cuda::BaseFieldVec, cuda::SecureFieldVec, CudaBackend};
use stwo_prover::core::backend::{Column, ColumnOps};
use stwo_prover::core::fields::{m31::BaseField, qm31::SecureField};

pub fn gpu_bit_reverse_base_field_iter(c: &mut Criterion) {
    const BITS: usize = 28;
    let size = 1 << BITS;
    let mut data = BaseFieldVec::from_vec((0..size).map(BaseField::from).collect_vec());

    c.bench_function(&format!("gpu bit_rev base_field {} bit single reference", BITS), |b| {
        b.iter(|| {
            <CudaBackend as ColumnOps<BaseField>>::bit_reverse_column(&mut data);
        })
    });
}

pub fn gpu_bit_reverse_secure_field(c: &mut Criterion) {
    const BITS: usize = 26;
    let size = 1 << BITS;

    let mut rng = SmallRng::seed_from_u64(0);
    let data = SecureFieldVec::from_vec((0..size).map(|_| rng.gen()).collect());
    assert_eq!(data.len(), size);

    c.bench_function(&format!("gpu bit_rev secure_field {} bit", BITS), |b| {
        b.iter_batched(||  data.clone(), 
            |mut data| <CudaBackend as ColumnOps<SecureField>>::bit_reverse_column(&mut data),
            BatchSize::PerIteration 
        );
    });
}

pub fn gpu_bit_reverse_base_field_with_large_drop(c: &mut Criterion) {
    const BITS: usize = 28;
    let size = 1 << BITS;
    let data = BaseFieldVec::from_vec((0..size).map(BaseField::from).collect_vec());

    c.bench_function(&format!("gpu bit_rev base_field with large drop {} bit", BITS), |b| {
        b.iter_with_large_drop(|| {
            let mut data = data.clone();
            move || {
                <CudaBackend as ColumnOps<BaseField>>::bit_reverse_column(&mut data)
            }
        });
    });
}

pub fn gpu_bit_reverse_base_field_iter_batched_dtd_copy(c: &mut Criterion) {
    const BITS: usize = 28;
    let size = 1 << BITS;
    let data = BaseFieldVec::from_vec((0..size).map(BaseField::from).collect_vec());

    c.bench_function(&format!("gpu bit_rev base_field {} bit multiple setup dtd", BITS), |b| {
        b.iter_batched(|| 
            data.clone(), 
            |mut data| <CudaBackend as ColumnOps<BaseField>>::bit_reverse_column(&mut data),
            BatchSize::PerIteration, 
        );
    });
}

pub fn gpu_bit_reverse_base_field_iter_batched_htd_copy(c: &mut Criterion) {
    const BITS: usize = 28;
    let size = 1 << BITS;
    //let data = BaseFieldVec::from_vec((0..size).map(BaseField::from).collect_vec());

    c.bench_function(&format!("gpu bit_rev base_field {} bit multiple setup htd", BITS), |b| {
        b.iter_batched(|| 
            BaseFieldVec::from_vec((0..size).map(BaseField::from).collect_vec()).clone(), 
            |mut data| <CudaBackend as ColumnOps<BaseField>>::bit_reverse_column(&mut data),
            BatchSize::PerIteration, 
        );
    });
}

pub fn gpu_bit_reverse_base_field_iter_initializing(c: &mut Criterion) {
    const BITS: usize = 28;
    let size = 1 << BITS;
    let vec = (0..size).map(BaseField::from).collect_vec();

    c.bench_function(&format!("gpu bit_rev base_field {} bit initializing", BITS), |b| {
        b.iter_with_setup(
            || vec.clone(),
            |cloned_vec| BaseFieldVec::from_vec(cloned_vec)
        )
    });
}

pub fn gpu_bit_reverse_base_field_iter_cloning(c: &mut Criterion) {
    const BITS: usize = 28;
    let size = 1 << BITS;
    let data = BaseFieldVec::from_vec((0..size).map(BaseField::from).collect_vec());

    c.bench_function(&format!("gpu bit_rev base_field {} bit cloning", BITS), |b| {
        b.iter(|| {
            let _ = data.clone();
            
        })
    });
}

criterion_group!(
    name = bit_reverse;
    config = Criterion::default().sample_size(10);
    targets = gpu_bit_reverse_base_field_iter_initializing,gpu_bit_reverse_base_field_iter_cloning); // gpu_bit_reverse_base_field_iter, gpu_bit_reverse_base_field_with_large_drop, gpu_bit_reverse_base_field_iter_batched_dtd_copy, gpu_bit_reverse_base_field_iter_batched_htd_copy); //, gpu_bit_reverse_secure_field);
criterion_main!(bit_reverse);
