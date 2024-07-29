use std::ops::Div;

use stwo_prover::core::air::accumulation::{
    ColumnAccumulator, DomainEvaluationAccumulator, PointEvaluationAccumulator,
};
use stwo_prover::core::air::mask::shifted_mask_points;
use stwo_prover::core::air::{Component, ComponentProver, ComponentTrace};
use stwo_prover::core::backend::cpu::CpuCircleEvaluation;
use stwo_prover::core::backend::{Column, CpuBackend};
use stwo_prover::core::circle::{CirclePoint, Coset};
use stwo_prover::core::constraints::{coset_vanishing, pair_vanishing};
use stwo_prover::core::fields::m31::BaseField;
use stwo_prover::core::fields::qm31::SecureField;
use stwo_prover::core::fields::secure_column::SecureColumnByCoords;
use stwo_prover::core::fields::{ExtensionOf, FieldExpOps};
use stwo_prover::core::pcs::TreeVec;
use stwo_prover::core::poly::circle::{CanonicCoset, CircleEvaluation};
use stwo_prover::core::poly::BitReversedOrder;
use stwo_prover::core::utils::bit_reverse_index;
use stwo_prover::core::{ColumnVec, InteractionElements, LookupValues};
use stwo_prover::trace_generation::registry::ComponentGenerationRegistry;
use stwo_prover::trace_generation::{ComponentGen, ComponentTraceGenerator, BASE_TRACE};

use crate::cuda::BaseFieldVec;
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

#[derive(Clone)]
pub struct FibonacciTraceGenerator {
    input: Option<FibonacciInput>,
}

impl ComponentGen for FibonacciTraceGenerator {}

impl FibonacciTraceGenerator {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self { input: None }
    }

    pub fn inputs_set(&self) -> bool {
        self.input.is_some()
    }
}

impl ComponentTraceGenerator<CudaBackend> for FibonacciTraceGenerator {
    type Component = FibonacciComponent;
    type Inputs = FibonacciInput;

    fn add_inputs(&mut self, inputs: &Self::Inputs) {
        todo!()
    }

    fn write_trace(
        component_id: &str,
        registry: &mut ComponentGenerationRegistry,
    ) -> ColumnVec<CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>> {
        todo!()
    }

    fn write_interaction_trace(
        &self,
        _trace: &ColumnVec<&CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>>,
        _elements: &InteractionElements,
    ) -> ColumnVec<CircleEvaluation<CudaBackend, BaseField, BitReversedOrder>> {
        vec![]
    }

    fn component(&self) -> Self::Component {
        todo!()
    }
}

impl ComponentProver<CudaBackend> for FibonacciComponent {
    fn evaluate_constraint_quotients_on_domain(
        &self,
        trace: &ComponentTrace<'_, CudaBackend>,
        evaluation_accumulator: &mut DomainEvaluationAccumulator<CudaBackend>,
        _interaction_elements: &InteractionElements,
        _lookup_values: &LookupValues,
    ) {
        let poly = &trace.polys[BASE_TRACE][0];
        let trace_domain = CanonicCoset::new(self.log_size);

        // `self.log_size + 1` porque el grado de las transiciones es 2.
        let trace_eval_domain = CanonicCoset::new(self.log_size + 1).circle_domain();
        let lde_trace_eval = poly.evaluate(trace_eval_domain).bit_reverse();
        let cpu_lde_trace_eval =
            CpuCircleEvaluation::new(lde_trace_eval.domain, lde_trace_eval.values.to_cpu());

        // Step constraint.
        let constraint_log_degree_bound = trace_domain.log_size() + 1;
        let [mut accum] = evaluation_accumulator.columns([(constraint_log_degree_bound, 2)]);
        let mut cpu_accum_col = accum.col.to_cpu();
        let mut cpu_accum = ColumnAccumulator::<CpuBackend> {
            random_coeff_powers: accum.random_coeff_powers.clone(),
            col: &mut cpu_accum_col,
        };
        let constraint_eval_domain = trace_eval_domain;

        for (off, point_coset) in [
            (0, constraint_eval_domain.half_coset),
            (
                constraint_eval_domain.half_coset.size(),
                constraint_eval_domain.half_coset.conjugate(),
            ),
        ] {
            // eval se trae el cacho de eval que corresponde a la primera o la segunda parte
            let eval =
                cpu_lde_trace_eval.fetch_eval_on_coset(point_coset.shift(trace_domain.index_at(0)));
            let mul = trace_domain.step_size().div(point_coset.step_size);
            for (i, point) in point_coset.iter().enumerate() {
                let mask = [eval[i], eval[i as isize + mul], eval[i as isize + 2 * mul]];
                let mut res = self.boundary_constraint_eval_quotient_by_mask(point, &[mask[0]])
                    * cpu_accum.random_coeff_powers[0];
                res += self.step_constraint_eval_quotient_by_mask(point, &mask)
                    * cpu_accum.random_coeff_powers[1];
                cpu_accum.accumulate(bit_reverse_index(i + off, constraint_log_degree_bound), res);
            }
        }

        *accum.col = SecureColumnByCoords::<CudaBackend> {
            columns: [
                BaseFieldVec::from_vec(cpu_accum.col.columns[0].clone()),
                BaseFieldVec::from_vec(cpu_accum.col.columns[1].clone()),
                BaseFieldVec::from_vec(cpu_accum.col.columns[2].clone()),
                BaseFieldVec::from_vec(cpu_accum.col.columns[3].clone()),
            ],
        };
    }

    fn lookup_values(&self, _trace: &ComponentTrace<'_, CudaBackend>) -> LookupValues {
        LookupValues::default()
    }
}
