use stwo_prover::core::{ColumnVec, InteractionElements, LookupValues};
use stwo_prover::core::air::{Component, ComponentProver, ComponentTrace};
use stwo_prover::core::air::accumulation::{
    DomainEvaluationAccumulator, PointEvaluationAccumulator,
};
use stwo_prover::core::air::mask::shifted_mask_points;
use stwo_prover::core::backend::Column;
use stwo_prover::core::circle::{CirclePoint, Coset};
use stwo_prover::core::constraints::{coset_vanishing, pair_vanishing};
use stwo_prover::core::fields::{ExtensionOf, FieldExpOps};
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::fields::qm31::SecureField;
use stwo_prover::core::pcs::TreeVec;
use stwo_prover::core::poly::circle::CanonicCoset;

use crate::cuda::{self};
use crate::CudaBackend;

#[derive(Clone)]
pub struct FibonacciComponent {
    pub log_size: u32,
    pub claim: BaseField,
}

impl FibonacciComponent {
    pub fn new(log_size: u32, claim: BaseField) -> Self {
        Self { log_size, claim }
    }

    /// Evaluates the step constraint quotient polynomial on a single point.
    /// The step constraint is defined as:
    ///   mask[0]^2 + mask[1]^2 - mask[2]
    fn step_constraint_eval_quotient_by_mask<F: ExtensionOf<BaseField>>(
        &self,
        point: CirclePoint<F>,
        mask: &[F; 3],
    ) -> F {
        let constraint_zero_domain = Coset::subgroup(self.log_size);
        let constraint_value = mask[0].square() + mask[1].square() - mask[2];
        let selector = pair_vanishing(
            constraint_zero_domain
                .at(constraint_zero_domain.size() - 2)
                .into_ef(),
            constraint_zero_domain
                .at(constraint_zero_domain.size() - 1)
                .into_ef(),
            point,
        );

        let num = constraint_value * selector;
        let denom = coset_vanishing(constraint_zero_domain, point);
        num / denom
    }

    /// Evaluates the boundary constraint quotient polynomial on a single point.
    fn boundary_constraint_eval_quotient_by_mask<F: ExtensionOf<BaseField>>(
        &self,
        point: CirclePoint<F>,
        mask: &[F; 1],
    ) -> F {
        let constraint_zero_domain = Coset::subgroup(self.log_size);
        let p = constraint_zero_domain.at(constraint_zero_domain.size() - 1);
        // On (1,0), we should get 1.
        // On p, we should get self.claim.
        // 1 + y * (self.claim - 1) * p.y^-1
        // TODO(spapini): Cache the constant.
        let linear = F::one() + point.y * (self.claim - BaseField::from(1)) * p.y.inverse();
        let num = mask[0] - linear;
        let denom = pair_vanishing(p.into_ef(), CirclePoint::zero(), point);
        num / denom
    }
}

impl Component for FibonacciComponent {
    fn n_constraints(&self) -> usize {
        2
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        // Step constraint is of degree 2.
        self.log_size + 1
    }

    fn trace_log_degree_bounds(&self) -> TreeVec<ColumnVec<u32>> {
        TreeVec::new(vec![vec![self.log_size]])
    }

    fn mask_points(
        &self,
        point: CirclePoint<SecureField>,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        TreeVec::new(vec![shifted_mask_points(
            &vec![vec![0, 1, 2]],
            &[CanonicCoset::new(self.log_size)],
            point,
        )])
    }

    fn evaluate_constraint_quotients_at_point(
        &self,
        point: CirclePoint<SecureField>,
        mask: &TreeVec<ColumnVec<Vec<SecureField>>>,
        evaluation_accumulator: &mut PointEvaluationAccumulator,
        _interaction_elements: &InteractionElements,
        _lookup_values: &LookupValues,
    ) {
        evaluation_accumulator.accumulate(
            self.step_constraint_eval_quotient_by_mask(point, &mask[0][0][..].try_into().unwrap()),
        );
        evaluation_accumulator.accumulate(self.boundary_constraint_eval_quotient_by_mask(
            point,
            &mask[0][0][..1].try_into().unwrap(),
        ));
    }
}

#[derive(Copy, Clone)]
pub struct FibonacciInput {
    pub log_size: u32,
    pub claim: BaseField,
}

impl ComponentProver<CudaBackend> for FibonacciComponent {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &ComponentTrace<'_, CudaBackend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<CudaBackend>,
        _interaction_elements: &InteractionElements,
        _lookup_values: &LookupValues,
    ) {
        let trace_domain = CanonicCoset::new(self.log_size);

        let trace_eval_domain = CanonicCoset::new(self.log_size + 1).circle_domain();
        let large_eval = trace.evals.0[0][0].clone().bit_reverse();
        assert_eq!(trace_domain.coset.step, trace_eval_domain.half_coset.initial().double().double());

        let constraint_log_degree_bound = trace_domain.log_size() + 1;
        let [accum] = evaluation_accumulator.columns([(constraint_log_degree_bound, 2)]);

        unsafe {
            cuda::bindings::fibonacci_component_evaluate_constraint_quotients_on_domain(
                large_eval.device_ptr,
                large_eval.len() as u32,
                accum.col.columns[0].device_ptr,
                accum.col.columns[1].device_ptr,
                accum.col.columns[2].device_ptr,
                accum.col.columns[3].device_ptr,
                self.claim,
                trace_eval_domain.half_coset.initial().into(),
                trace_eval_domain.half_coset.step.into(),
                accum.random_coeff_powers[0].into(),
                accum.random_coeff_powers[1].into(),
            );
        }
    }

    fn lookup_values(&self, _trace: &ComponentTrace<'_, CudaBackend>) -> LookupValues {
        LookupValues::default()
    }
}
