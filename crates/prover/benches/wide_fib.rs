#![feature(iter_array_chunks)]

use criterion::{criterion_group, criterion_main, Criterion};
use stwo_prover::core::backend::simd::SimdBackend;
use stwo_prover::core::channel::{Blake2sChannel, Channel};
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::fields::IntoSlice;
use stwo_prover::core::prover::{prove, verify};
use stwo_prover::core::vcs::blake2_hash::Blake2sHasher;
use stwo_prover::core::vcs::hasher::Hasher;
use stwo_prover::examples::wide_fibonacci::component::{WideFibAir, WideFibComponent};
use stwo_prover::examples::wide_fibonacci::simd::gen_trace;
use tracing::{span, Level};

const LOG_N_ROWS: u32 = 16;

const LOG_N_COLS: u32 = 8;

fn test_simd_wide_fib_prove() {
    // const LOG_N_ROWS: u32 = 16;
    let component = WideFibComponent {
        log_fibonacci_size: LOG_N_COLS as u32,
        log_n_instances: LOG_N_ROWS,
    };
    let span = span!(Level::INFO, "Trace generation").entered();
    let trace = gen_trace(component.log_column_size());
    span.exit();
    let channel = &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&[])));
    let air = WideFibAir { component };
    let proof = prove::<SimdBackend>(&air, channel, trace).unwrap();

    let channel = &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&[])));
    verify(proof, &air, channel).unwrap();
}

fn bench_wide_fib_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("wide fib simd throughput");
    group.bench_function(&format!("simd wide fib"), |b| {
        b.iter_with_large_drop(|| test_simd_wide_fib_prove())
    });
}

fn wide_fib_simd_benches(c: &mut Criterion) {
    bench_wide_fib_simd(c);
}

criterion_group!(
    name = benches;
    config = Criterion::default()
            .sample_size(10)
            .measurement_time(std::time::Duration::new(24,0));
    targets = wide_fib_simd_benches);
criterion_main!(benches);
