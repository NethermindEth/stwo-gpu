use stwo_prover::core::channel::{Blake2sChannel, Channel};
use stwo_prover::core::fields::{FieldExpOps, IntoSlice};
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::poly::BitReversedOrder;
use stwo_prover::core::poly::circle::{CanonicCoset, CircleEvaluation};
use stwo_prover::core::prover::{ProvingError, StarkProof, VerificationError};
use stwo_prover::core::vcs::blake2_hash::Blake2sHasher;
use stwo_prover::core::vcs::hasher::Hasher;
use stwo_prover::trace_generation::{commit_and_prove, commit_and_verify};

use crate::cuda::BaseFieldVec;
use crate::CudaBackend;

use self::air::FibonacciAir;
use self::component::FibonacciComponent;

pub mod air;
mod component;

#[derive(Clone)]
pub struct Fibonacci {
    pub air: FibonacciAir,
}

impl Fibonacci {
    pub fn new(log_size: u32, claim: BaseField) -> Self {
        let component = FibonacciComponent::new(log_size, claim);
        Self {
            air: FibonacciAir::new(component),
        }
    }

    pub fn get_trace(&self) -> CircleEvaluation<CudaBackend, BaseField, BitReversedOrder> {
        // Trace.
        let trace_domain = CanonicCoset::new(self.air.component.log_size);
        let mut trace = Vec::with_capacity(trace_domain.size());

        // Fill trace with fibonacci squared.
        let mut a = BaseField::from(1);
        let mut b = BaseField::from(1);
        for _ in 0..trace_domain.size() {
            trace.push(a);
            let tmp = a.square() + b.square();
            a = b;
            b = tmp;
        }

        let trace = BaseFieldVec::from_vec(trace);

        // Returns as a CircleEvaluation.
        CircleEvaluation::new_canonical_ordered(trace_domain, trace)
    }

    pub fn prove(&self) -> Result<StarkProof, ProvingError> {
        let trace = self.get_trace();
        // println!("trace: {:?}", trace.values.to_cpu());
        let channel = &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&[self
            .air
            .component
            .claim])));
        commit_and_prove(&self.air, channel, vec![trace])
    }

    pub fn verify(&self, proof: StarkProof) -> Result<(), VerificationError> {
        let channel = &mut Blake2sChannel::new(Blake2sHasher::hash(BaseField::into_slice(&[self
            .air
            .component
            .claim])));
        commit_and_verify(proof, &self.air, channel)
    }
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use itertools::Itertools;
    use stwo_prover::core::{InteractionElements, LookupValues};
    use stwo_prover::core::air::{AirExt, AirProverExt, Component, ComponentTrace};
    use stwo_prover::core::air::accumulation::PointEvaluationAccumulator;
    use stwo_prover::core::circle::CirclePoint;
    use stwo_prover::core::fields::m31::BaseField;
    use stwo_prover::core::fields::qm31::SecureField;
    use stwo_prover::core::pcs::TreeVec;
    use stwo_prover::core::poly::circle::CanonicCoset;
    use stwo_prover::trace_generation::BASE_TRACE;

    use crate::examples::Fibonacci;

    #[test]
    fn test_composition_polynomial_is_low_degree() {
        let fib = Fibonacci::new(5, BaseField::from(443693538));
        let trace = fib.get_trace();
        let trace_poly = trace.interpolate();
        let trace_eval =
            trace_poly.evaluate(CanonicCoset::new(trace_poly.log_size() + 1).circle_domain());
        let trace = ComponentTrace::new(
            TreeVec::new(vec![vec![&trace_poly]]),
            TreeVec::new(vec![vec![&trace_eval]]),
        );

        let random_coeff = SecureField::from_m31(
            BaseField::from(2213980),
            BaseField::from(2213981),
            BaseField::from(2213982),
            BaseField::from(2213983),
        );
        let component_traces = vec![trace];
        let composition_polynomial_poly = fib.air.compute_composition_polynomial(
            random_coeff,
            &component_traces,
            &InteractionElements::default(),
            &LookupValues::default(),
        );

        // Evaluate this polynomial at another point out of the evaluation domain and compare to
        // what we expect.
        let point = CirclePoint::<SecureField>::get_point(98989892);

        let points = fib.air.mask_points(point);
        let mask_values = zip(&component_traces[0].polys[BASE_TRACE], &points[0])
            .map(|(poly, points)| {
                points
                    .iter()
                    .map(|point| poly.eval_at_point(*point))
                    .collect_vec()
            })
            .collect_vec();

        let mut evaluation_accumulator = PointEvaluationAccumulator::new(random_coeff);
        fib.air.component.evaluate_constraint_quotients_at_point(
            point,
            &TreeVec::new(vec![mask_values]),
            &mut evaluation_accumulator,
            &InteractionElements::default(),
            &LookupValues::default(),
        );
        let oods_value = evaluation_accumulator.finalize();

        assert_eq!(oods_value, composition_polynomial_poly.eval_at_point(point));
    }

    #[test]
    fn test_fib_prove() {
        const FIB_LOG_SIZE: u32 = 5;
        let fib = Fibonacci::new(FIB_LOG_SIZE, BaseField::from(443693538));

        let proof = fib.prove().unwrap();
        fib.verify(proof).unwrap();
    }

    #[should_panic]
    #[test]
    fn test_prove_invalid_trace_value() {
        const FIB_LOG_SIZE: u32 = 5;
        let fib = Fibonacci::new(FIB_LOG_SIZE, BaseField::from(443693538));

        let mut invalid_proof = fib.prove().unwrap();
        invalid_proof.commitment_scheme_proof.queried_values.0[BASE_TRACE][0][3] +=
            BaseField::from(1);

        let _ = fib.verify(invalid_proof).unwrap();
    }

    //     #[test]
    //     fn test_prove_invalid_trace_oods_values() {
    //         const FIB_LOG_SIZE: u32 = 5;
    //         let fib = Fibonacci::new(FIB_LOG_SIZE, m31!(443693538));

    //         let mut invalid_proof = fib.prove().unwrap();
    //         invalid_proof
    //             .commitment_scheme_proof
    //             .sampled_values
    //             .swap(0, 1);

    //         let error = fib.verify(invalid_proof).unwrap_err();
    //         assert_matches!(error, VerificationError::InvalidStructure(_));
    //         assert!(error
    //             .to_string()
    //             .contains("Unexpected sampled_values structure"));
    //     }

    //     #[test]
    //     fn test_prove_insufficient_trace_values() {
    //         const FIB_LOG_SIZE: u32 = 5;
    //         let fib = Fibonacci::new(FIB_LOG_SIZE, m31!(443693538));

    //         let mut invalid_proof = fib.prove().unwrap();
    //         invalid_proof.commitment_scheme_proof.queried_values.0[BASE_TRACE][0].pop();

    //         let error = fib.verify(invalid_proof).unwrap_err();
    //         assert_matches!(error, VerificationError::Merkle(_));
    //     }

    //     #[test]
    //     fn test_rectangular_multi_fibonacci() {
    //         let multi_fib = MultiFibonacci::new(vec![5; 16], vec![m31!(443693538); 16]);
    //         let proof = multi_fib.prove().unwrap();
    //         multi_fib.verify(proof).unwrap();
    //     }

    //     #[test]
    //     fn test_mixed_degree_multi_fibonacci() {
    //         let multi_fib = MultiFibonacci::new(
    //             // TODO(spapini): Change order of log_sizes.
    //             vec![3, 5, 7],
    //             vec![m31!(1056169651), m31!(443693538), m31!(722122436)],
    //         );
    //         let proof = multi_fib.prove().unwrap();
    //         multi_fib.verify(proof).unwrap();
    //     }
}
