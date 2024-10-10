mod wide_fibonacci;

use itertools::Itertools;

use stwo_prover::constraint_framework::{EvalAtRow, FrameworkComponent, FrameworkEval};
use stwo_prover::core::air::accumulation::DomainEvaluationAccumulator;
use stwo_prover::core::air::{ComponentProver, Trace};
use stwo_prover::core::backend::simd::column::BaseColumn;
use stwo_prover::core::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
use stwo_prover::core::backend::simd::SimdBackend;
use stwo_prover::core::backend::{Col, Column};
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::fields::secure_column::SecureColumnByCoords;
use stwo_prover::core::fields::FieldExpOps;
use stwo_prover::core::pcs::TreeVec;
use stwo_prover::core::poly::circle::{CanonicCoset, CircleEvaluation, CirclePoly};
use stwo_prover::core::poly::BitReversedOrder;
use stwo_prover::core::ColumnVec;

use crate::cuda::BaseFieldVec;
use crate::CudaBackend;

pub type WideFibonacciComponentCuda<const N: usize> = FrameworkComponent<WideFibonacciEvalCuda<N>>;

pub struct FibInput {
    a: PackedBaseField,
    b: PackedBaseField,
}

/// A component that enforces the Fibonacci sequence.
/// Each row contains a seperate Fibonacci sequence of length `N`.
#[derive(Clone)]
pub struct WideFibonacciEvalCuda<const N: usize> {
    pub log_n_rows: u32,
}
impl<const N: usize> FrameworkEval for WideFibonacciEvalCuda<N> {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let mut a = eval.next_trace_mask();
        let mut b = eval.next_trace_mask();
        for _ in 2..N {
            let c = eval.next_trace_mask();
            eval.add_constraint(c - (a.square() + b.square()));
            a = b;
            b = c;
        }
        eval
    }
}

pub fn generate_trace<const N: usize>(
    log_size: u32,
    inputs: &[FibInput],
) -> ColumnVec<CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>> {
    assert!(log_size >= LOG_N_LANES);
    assert_eq!(inputs.len(), 1 << (log_size - LOG_N_LANES));
    let mut trace = (0..N)
        .map(|_| Col::<SimdBackend, BaseField>::zeros(1 << log_size))
        .collect_vec();
    for (vec_index, input) in inputs.iter().enumerate() {
        let mut a = input.a;
        let mut b = input.b;
        trace[0].data[vec_index] = a;
        trace[1].data[vec_index] = b;
        trace.iter_mut().skip(2).for_each(|col| {
            (a, b) = (b, a.square() + b.square());
            col.data[vec_index] = b;
        });
    }
    let domain = CanonicCoset::new(log_size).circle_domain();
    trace
        .into_iter()
        .map(|eval| {
            let eval = BaseFieldVec::from_vec(eval.to_cpu());
            CircleEvaluation::<CudaBackend, _, BitReversedOrder>::new(domain, eval)
        })
        .collect_vec()
}

impl<E: FrameworkEval> ComponentProver<CudaBackend> for FrameworkComponent<E> {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &Trace<'_, CudaBackend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<CudaBackend>,
    ) {
        let simd_polys = trace
            .polys
            .iter()
            .map(|column_vec| {
                column_vec
                    .iter()
                    .map(|circle_poly| {
                        CirclePoly::<SimdBackend>::new(BaseColumn::from_iter(
                            circle_poly.coeffs.to_cpu(),
                        ))
                    })
                    .collect_vec()
            })
            .collect_vec();

        let simd_evals = trace
            .evals
            .iter()
            .map(|column_vec| {
                column_vec
                    .iter()
                    .map(|circle_eval| {
                        CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(
                            circle_eval.domain,
                            BaseColumn::from_iter(circle_eval.values.to_cpu()),
                        )
                    })
                    .collect_vec()
            })
            .collect_vec();

        let simd_trace = Trace::<'_, SimdBackend> {
            polys: TreeVec(
                simd_polys
                    .iter()
                    .map(|x| x.iter().collect_vec())
                    .collect_vec(),
            ),
            evals: TreeVec(
                simd_evals
                    .iter()
                    .map(|x| x.iter().collect_vec())
                    .collect_vec(),
            ),
        };

        let simd_sub_accumulations: Vec<Option<SecureColumnByCoords<SimdBackend>>> =
            evaluation_accumulator
                .sub_accumulations
                .iter()
                .map(|item| {
                    item.clone()
                        .map(|secure_column_by_coords| SecureColumnByCoords {
                            columns: [
                                BaseColumn::from_iter(secure_column_by_coords.columns[0].to_cpu()),
                                BaseColumn::from_iter(secure_column_by_coords.columns[1].to_cpu()),
                                BaseColumn::from_iter(secure_column_by_coords.columns[2].to_cpu()),
                                BaseColumn::from_iter(secure_column_by_coords.columns[3].to_cpu()),
                            ],
                        })
                })
                .collect_vec();
        let mut simd_evaluation_accumulator = DomainEvaluationAccumulator::<SimdBackend> {
            random_coeff_powers: evaluation_accumulator.random_coeff_powers.clone(),
            sub_accumulations: simd_sub_accumulations,
        };
        <Self as ComponentProver<SimdBackend>>::evaluate_constraint_quotients_on_domain(
            self,
            &simd_trace,
            &mut simd_evaluation_accumulator,
        );
        evaluation_accumulator.random_coeff_powers =
            simd_evaluation_accumulator.random_coeff_powers;
        evaluation_accumulator.sub_accumulations = simd_evaluation_accumulator
            .sub_accumulations
            .into_iter()
            .map(|item| {
                item.map(|secure_column_by_coords| SecureColumnByCoords {
                    columns: [
                        BaseFieldVec::from_vec(secure_column_by_coords.columns[0].to_cpu()),
                        BaseFieldVec::from_vec(secure_column_by_coords.columns[1].to_cpu()),
                        BaseFieldVec::from_vec(secure_column_by_coords.columns[2].to_cpu()),
                        BaseFieldVec::from_vec(secure_column_by_coords.columns[3].to_cpu()),
                    ],
                })
            })
            .collect_vec();
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use num_traits::One;

    use crate::examples::WideFibonacciComponentCuda;
    use crate::CudaBackend;

    use super::{generate_trace, FibInput, WideFibonacciEvalCuda};
    use stwo_prover::constraint_framework::{
        assert_constraints, AssertEvaluator, FrameworkEval, TraceLocationAllocator,
    };
    use stwo_prover::core::air::Component;
    use stwo_prover::core::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
    use stwo_prover::core::backend::Column;
    use stwo_prover::core::channel::Blake2sChannel;
    use stwo_prover::core::fields::m31::BaseField;
    use stwo_prover::core::pcs::{
        CommitmentSchemeProver, CommitmentSchemeVerifier, PcsConfig, TreeVec,
    };
    use stwo_prover::core::poly::circle::{CanonicCoset, CircleEvaluation, PolyOps};
    use stwo_prover::core::poly::BitReversedOrder;
    use stwo_prover::core::prover::{prove, verify};
    use stwo_prover::core::vcs::blake2_merkle::Blake2sMerkleChannel;
    use stwo_prover::core::ColumnVec;

    const FIB_SEQUENCE_LENGTH: usize = 1024;

    fn generate_test_trace(
        log_n_instances: u32,
    ) -> ColumnVec<CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>> {
        let inputs = (0..(1 << (log_n_instances - LOG_N_LANES)))
            .map(|i| FibInput {
                a: PackedBaseField::one(),
                b: PackedBaseField::from_array(std::array::from_fn(|j| {
                    BaseField::from_u32_unchecked((i * 16 + j) as u32)
                })),
            })
            .collect_vec();
        generate_trace::<FIB_SEQUENCE_LENGTH>(log_n_instances, &inputs)
    }

    // TODO: Uncomment tests
    fn fibonacci_constraint_evaluator<const N: u32>(eval: AssertEvaluator<'_>) {
        WideFibonacciEvalCuda::<FIB_SEQUENCE_LENGTH> { log_n_rows: N }.evaluate(eval);
    }

    #[test]
    fn test_wide_fibonacci_constraints() {
        const LOG_N_INSTANCES: u32 = 6;
        let traces = TreeVec::new(vec![generate_test_trace(LOG_N_INSTANCES)]);
        let trace_polys =
            traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect_vec());

        assert_constraints(
            &trace_polys,
            CanonicCoset::new(LOG_N_INSTANCES),
            fibonacci_constraint_evaluator::<LOG_N_INSTANCES>,
        );
    }

    #[test]
    #[should_panic]
    fn test_wide_fibonacci_constraints_fails() {
        const LOG_N_INSTANCES: u32 = 9;

        let mut trace = generate_test_trace(LOG_N_INSTANCES);
        // Modify the trace such that a constraint fail.
        trace[17].values.set(2, BaseField::one());
        let traces = TreeVec::new(vec![trace]);
        let trace_polys =
            traces.map(|trace| trace.into_iter().map(|c| c.interpolate()).collect_vec());

        assert_constraints(
            &trace_polys,
            CanonicCoset::new(LOG_N_INSTANCES),
            fibonacci_constraint_evaluator::<LOG_N_INSTANCES>,
        );
    }

    #[test_log::test]
    fn test_wide_fib_prove() {
        const LOG_N_INSTANCES: u32 = 16;
        let config = PcsConfig::default();
        // Precompute twiddles.
        let twiddles = CudaBackend::precompute_twiddles(
            CanonicCoset::new(LOG_N_INSTANCES + 1 + config.fri_config.log_blowup_factor)
                .circle_domain()
                .half_coset,
        );

        // Setup protocol.
        let prover_channel = &mut Blake2sChannel::default();
        let commitment_scheme =
            &mut CommitmentSchemeProver::<CudaBackend, Blake2sMerkleChannel>::new(
                config, &twiddles,
            );

        // Trace.
        let trace = generate_test_trace(LOG_N_INSTANCES);
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(trace);
        tree_builder.commit(prover_channel);

        // Prove constraints.
        let component = WideFibonacciComponentCuda::new(
            &mut TraceLocationAllocator::default(),
            WideFibonacciEvalCuda::<FIB_SEQUENCE_LENGTH> {
                log_n_rows: LOG_N_INSTANCES,
            },
        );

        let proof = prove::<CudaBackend, Blake2sMerkleChannel>(
            &[&component],
            prover_channel,
            commitment_scheme,
        )
        .unwrap();

        // Verify.
        let verifier_channel = &mut Blake2sChannel::default();
        let commitment_scheme = &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);

        // Retrieve the expected column sizes in each commitment interaction, from the AIR.
        let sizes = component.trace_log_degree_bounds();
        commitment_scheme.commit(proof.commitments[0], &sizes[0], verifier_channel);
        verify(&[&component], verifier_channel, commitment_scheme, proof).unwrap();
    }
}
