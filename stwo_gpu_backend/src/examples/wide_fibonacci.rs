use itertools::Itertools;

use num_traits::Zero;
use stwo_prover::core::air::mask::fixed_mask_points;
use stwo_prover::core::air::accumulation::{DomainEvaluationAccumulator, PointEvaluationAccumulator};
use stwo_prover::core::air::{Air, Component, ComponentProver, Trace};
use stwo_prover::core::backend::CpuBackend;
use stwo_prover::core::circle::CirclePoint;
use stwo_prover::core::constraints::coset_vanishing;
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::fields::qm31::SecureField;
use stwo_prover::core::fields::FieldExpOps;
use stwo_prover::core::pcs::TreeVec;
use stwo_prover::core::poly::circle::{CanonicCoset, CircleEvaluation};
use stwo_prover::core::poly::BitReversedOrder;
use stwo_prover::core::utils::bit_reverse;
use stwo_prover::core::{ColumnVec};

pub const LOG_N_COLUMNS: usize = 16;
pub const N_COLUMNS: usize = 1 << LOG_N_COLUMNS;

const ALPHA_ID: &str = "wide_fibonacci_alpha";
const Z_ID: &str = "wide_fibonacci_z";

/// Component that computes 2^`self.log_n_instances` instances of fibonacci sequences of size
/// 2^`self.log_fibonacci_size`. The numbers are computes over [N_COLUMNS] trace columns. The
/// number of rows (i.e the size of the columns) is determined by the parameters above (see
/// [WideFibComponent::log_column_size()]).
pub struct WideFibComponent {
    pub log_fibonacci_size: u32,
    pub log_n_instances: u32,
}

impl WideFibComponent {
    /// Returns the log of the size of the columns in the trace (which could also be looked at as
    /// the log number of rows).
    pub fn log_column_size(&self) -> u32 {
        self.log_n_instances as u32
    }

    pub fn log_n_columns(&self) -> usize {
        LOG_N_COLUMNS
    }

    pub fn n_columns(&self) -> usize {
        N_COLUMNS
    }
}

pub struct WideFibAir {
    pub component: WideFibComponent,
}

impl Air for WideFibAir {
    fn components(&self) -> Vec<&dyn Component> {
        vec![&self.component]
    }
}

impl Component for WideFibComponent {
    fn n_constraints(&self) -> usize {
        self.n_columns() - 2
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_column_size() + 1
    }

    fn trace_log_degree_bounds(&self) -> TreeVec<Vec<u32>> {
        TreeVec(vec![vec![self.log_column_size(); self.n_columns()]])
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        TreeVec(vec![fixed_mask_points(&vec![vec![0_usize]; self.n_columns()], point)])
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &TreeVec<ColumnVec<Vec<SecureField>>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator
    ) {
        let constraint_zero_domain = CanonicCoset::new(self.log_column_size()).coset;
        let denom = coset_vanishing(constraint_zero_domain, point);
        let denom_inverse = denom.inverse();
        for i in 0..self.n_columns() - 2 {
            let numerator = mask.0[0][i][0].square() + mask.0[0][i + 1][0].square() - mask.0[0][i + 2][0];
            evaluation_accumulator.accumulate(numerator * denom_inverse);
        }
    }
}

// Input for the fibonacci claim.
#[derive(Debug, Clone, Copy)]
pub struct Input {
    pub a: BaseField,
    pub b: BaseField,
}


impl ComponentProver<CpuBackend> for WideFibComponent {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &Trace<'_, CpuBackend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<CpuBackend>
    ) {
        let max_constraint_degree = self.max_constraint_log_degree_bound();
        let trace_eval_domain = CanonicCoset::new(max_constraint_degree).circle_domain();
        let trace_evals = &trace.evals;
        let zero_domain = CanonicCoset::new(self.log_column_size()).coset;
        let mut denoms = vec![];
        for point in trace_eval_domain.iter() {
            denoms.push(coset_vanishing(zero_domain, point));
        }
        bit_reverse(&mut denoms);
        let mut denom_inverses = vec![BaseField::zero(); 1 << (max_constraint_degree)];
        BaseField::batch_inverse(&denoms, &mut denom_inverses);
        let mut numerators = vec![SecureField::zero(); 1 << (max_constraint_degree)];
        let [mut accum] =
            evaluation_accumulator.columns([(max_constraint_degree, self.n_constraints())]);

        #[allow(clippy::needless_range_loop)]
        for i in 0..trace_eval_domain.size() {
            // Step constraints.
            for j in 0..self.n_columns() - 2 {
                numerators[i] += accum.random_coeff_powers[self.n_columns() - 3 - j]
                    * (trace_evals[0][j][i].square() + trace_evals[0][j + 1][i].square()
                        - trace_evals[0][j + 2][i]);
            }
        }
        for (i, (num, denom)) in numerators.iter().zip(denom_inverses.iter()).enumerate() {
            accum.accumulate(i, *num * *denom);
        }
    }
}

mod test{
    use itertools::Itertools;
    use num_traits::{One, Zero};
    use stwo_prover::core::{air::{Air, Component}, backend::{cpu::CpuCircleEvaluation, simd::m31::LOG_N_LANES, CpuBackend}, channel::Blake2sChannel, fields::{m31::BaseField, FieldExpOps, IntoSlice}, pcs::{CommitmentSchemeProver, CommitmentSchemeVerifier, PcsConfig}, poly::{circle::{CanonicCoset, CircleEvaluation, PolyOps}, BitReversedOrder}, prover::{prove, verify}, vcs::{blake2_hash::Blake2sHasher, blake2_merkle::Blake2sMerkleChannel}, ColumnVec};

    use crate::examples::{wide_fibonacci::{WideFibAir, LOG_N_COLUMNS}};

    use super::{Input, WideFibComponent, N_COLUMNS};

    /// Generates the trace for the wide Fibonacci example.
    fn generate_test_trace(
        log_n_instances: u32,
    ) -> ColumnVec<CircleEvaluation<CpuBackend, BaseField, BitReversedOrder>> {
        let inputs: Vec<Input> = (0..1<<log_n_instances).map( |i|
            Input {
                a: BaseField::one(),
                b: BaseField::from(i),
            }
        ).collect();

        let mut trace = vec![vec![BaseField::zero(); inputs.len()]; N_COLUMNS];
        for (index, input) in inputs.iter().enumerate() {
            write_trace_row(&mut trace, input, index);
        }

        let domain = CanonicCoset::new(log_n_instances).circle_domain();
        trace.into_iter().map(|column| {
            CircleEvaluation::<_, _, BitReversedOrder>::new(domain, column)
        }).collect_vec()
    }

    /// Writes the trace row for the wide Fibonacci example to dst, given a private input. Returns the
    /// last two elements of the row in case the sequence is continued.
    pub fn write_trace_row(
        dst: &mut [Vec<BaseField>],
        private_input: &Input,
        row_index: usize,
    ) {
        let n_columns = dst.len();
        dst[0][row_index] = private_input.a;
        dst[1][row_index] = private_input.b;
        for i in 2..n_columns {
            dst[i][row_index] = dst[i - 1][row_index].square() + dst[i - 2][row_index].square();
        }
    }

    #[test_log::test]
    fn test_single_instance_wide_fib_prove() {
        // Note: To see time measurement, run test with
        //   RUST_LOG_SPAN_EVENTS=enter,close RUST_LOG=info RUST_BACKTRACE=1 cargo test 
        //   test_single_instance_wide_fib_prove -- --nocapture
        const LOG_N_INSTANCES: u32 = 10;

        let config = PcsConfig::default();

        // Precompute twiddles.
        let twiddles = CpuBackend::precompute_twiddles(
            CanonicCoset::new(LOG_N_INSTANCES + 1 + config.fri_config.log_blowup_factor)
                .circle_domain()
                .half_coset,
        );

        // Setup protocol.
        let prover_channel = &mut Blake2sChannel::default();
        let commitment_scheme = &mut CommitmentSchemeProver::<CpuBackend, Blake2sMerkleChannel>::new(
                config, &twiddles,
        );

        // Trace.
        let trace = generate_test_trace(LOG_N_INSTANCES);
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_evals(trace);
        tree_builder.commit(prover_channel);
        
        let component = WideFibComponent {
            log_fibonacci_size: LOG_N_COLUMNS as u32,
            log_n_instances: LOG_N_INSTANCES,
        };
        
        let proof = prove::<CpuBackend, _>(&[&component], prover_channel, commitment_scheme).unwrap();

        // Verify.
        let verifier_channel = &mut Blake2sChannel::default();
        let commitment_scheme = &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);

        // Retrieve the expected column sizes in each commitment interaction, from the AIR.
        let sizes = component.trace_log_degree_bounds();
        commitment_scheme.commit(proof.commitments[0], &sizes[0], verifier_channel);
        verify(&[&component], verifier_channel, commitment_scheme, proof).unwrap();
    }
}
