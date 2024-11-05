use itertools::Itertools;
use stwo_prover::core::air::accumulation::{
    DomainEvaluationAccumulator, PointEvaluationAccumulator,
};
use stwo_prover::core::air::mask::fixed_mask_points;
use stwo_prover::core::air::{Component, ComponentProver, Trace};
use stwo_prover::core::circle::CirclePoint;
use stwo_prover::core::constraints::coset_vanishing;
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::fields::qm31::SecureField;
use stwo_prover::core::fields::FieldExpOps;
use stwo_prover::core::pcs::TreeVec;
use stwo_prover::core::poly::circle::CanonicCoset;
use stwo_prover::core::ColumnVec;

use crate::cuda::{bindings, BaseFieldVec, SecureFieldVec};
use crate::CudaBackend;

pub const LOG_N_COLUMNS: usize = 10;
pub const N_COLUMNS: usize = 1 << LOG_N_COLUMNS;

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

    pub fn n_columns(&self) -> usize {
        N_COLUMNS
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
        TreeVec(vec![fixed_mask_points(
            &vec![vec![0_usize]; self.n_columns()],
            point,
        )])
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &TreeVec<ColumnVec<Vec<SecureField>>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
    ) {
        let constraint_zero_domain = CanonicCoset::new(self.log_column_size()).coset;
        let denom = coset_vanishing(constraint_zero_domain, point);
        let denom_inverse = denom.inverse();
        for i in 0..self.n_columns() - 2 {
            let numerator =
                mask.0[0][i][0].square() + mask.0[0][i + 1][0].square() - mask.0[0][i + 2][0];
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

impl ComponentProver<CudaBackend> for WideFibComponent {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &Trace<'_, CudaBackend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<CudaBackend>,
    ) {
        let ext_domain_log_size = self.max_constraint_log_degree_bound();
        let ext_domain = CanonicCoset::new(ext_domain_log_size).circle_domain();
        let trace_evals_ext_domain = &trace.evals[0];
        let zero_domain = CanonicCoset::new(self.log_column_size()).coset;

        let mut denominators = vec![];
        for point in ext_domain.iter() {
            denominators.push(coset_vanishing(zero_domain, point));
        }

        let gpu_denominators = BaseFieldVec::from_vec(denominators);
        unsafe {
            bindings::bit_reverse_base_field(gpu_denominators.device_ptr, ext_domain.size());
        }
        let denominator_inverses = BaseFieldVec::new_zeroes(ext_domain.size());
        unsafe {
            bindings::batch_inverse_base_field(
                gpu_denominators.device_ptr,
                denominator_inverses.device_ptr,
                ext_domain.size(),
            );
        }

        let [column_accumulator] =
            evaluation_accumulator.columns([(ext_domain_log_size, self.n_constraints())]);
        let trace_evaluations_vec = trace_evals_ext_domain
            .iter()
            .map(|column_evaluations| column_evaluations.device_ptr)
            .collect_vec();
        let random_coeff_powers = SecureFieldVec::from_vec(column_accumulator.random_coeff_powers);

        unsafe {
            bindings::evaluate_wide_fibonacci_constraint_quotients_on_domain(
                column_accumulator.col.columns[0].device_ptr,
                column_accumulator.col.columns[1].device_ptr,
                column_accumulator.col.columns[2].device_ptr,
                column_accumulator.col.columns[3].device_ptr,
                trace_evaluations_vec.as_ptr(),
                random_coeff_powers.device_ptr,
                denominator_inverses.device_ptr,
                ext_domain.size() as u32,
                self.n_columns() as u32,
            );
        };
    }
}

mod test {
    use itertools::Itertools;
    use num_traits::{One, Zero};
    use stwo_prover::core::{
        air::Component,
        channel::Blake2sChannel,
        fields::{m31::BaseField, FieldExpOps},
        pcs::{CommitmentSchemeProver, CommitmentSchemeVerifier, PcsConfig},
        poly::{
            circle::{CanonicCoset, CircleEvaluation, PolyOps},
            BitReversedOrder,
        },
        prover::{prove, verify},
        vcs::blake2_merkle::Blake2sMerkleChannel,
        ColumnVec,
    };

    use crate::{cuda::BaseFieldVec, examples::wide_fibonacci::LOG_N_COLUMNS, CudaBackend};

    use super::{Input, WideFibComponent, N_COLUMNS};

    /// Generates the trace for the wide Fibonacci example.
    fn generate_test_trace(
        log_n_instances: u32,
    ) -> ColumnVec<CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>> {
        let inputs: Vec<Input> = (0..1 << log_n_instances)
            .map(|i| Input {
                a: BaseField::one(),
                b: BaseField::from(i),
            })
            .collect();

        let mut trace = vec![vec![BaseField::zero(); inputs.len()]; N_COLUMNS];
        for (index, input) in inputs.iter().enumerate() {
            write_trace_row(&mut trace, input, index);
        }

        let domain = CanonicCoset::new(log_n_instances).circle_domain();
        trace
            .into_iter()
            .map(|column| {
                CircleEvaluation::<_, _, BitReversedOrder>::new(
                    domain,
                    BaseFieldVec::from_vec(column),
                )
            })
            .collect_vec()
    }

    /// Writes the trace row for the wide Fibonacci example to dst, given a private input. Returns the
    /// last two elements of the row in case the sequence is continued.
    pub fn write_trace_row(dst: &mut [Vec<BaseField>], private_input: &Input, row_index: usize) {
        let n_columns = dst.len();
        dst[0][row_index] = private_input.a;
        dst[1][row_index] = private_input.b;
        for i in 2..n_columns {
            dst[i][row_index] = dst[i - 1][row_index].square() + dst[i - 2][row_index].square();
        }
    }

    #[test_log::test]
    fn test_cuda_constraints_for_wide_fib_prove() {
        // Note: To see time measurement, run test with
        //   RUST_LOG_SPAN_EVENTS=enter,close RUST_LOG=info RUST_BACKTRACE=1
        //   RUSTFLAGS="-Awarnings -C target-cpu=native -C target-feature=+avx2 -C opt-level=3"
        //   cargo test test_cuda_constraints_for_wide_fib_prove -- --nocapture
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

        let component = WideFibComponent {
            log_fibonacci_size: LOG_N_COLUMNS as u32,
            log_n_instances: LOG_N_INSTANCES,
        };

        let proof =
            prove::<CudaBackend, _>(&[&component], prover_channel, commitment_scheme).unwrap();

        // Verify.
        let verifier_channel = &mut Blake2sChannel::default();
        let commitment_scheme = &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);

        let sizes = component.trace_log_degree_bounds();
        commitment_scheme.commit(proof.commitments[0], &sizes[0], verifier_channel);
        verify(&[&component], verifier_channel, commitment_scheme, proof).unwrap();
    }
}
