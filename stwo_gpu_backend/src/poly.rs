use crate::cuda::bindings::CudaSecureField;
use crate::{
    backend::CudaBackend,
    cuda::{self},
};
use itertools::Itertools;
use stwo_prover::core::pcs::quotients::PointSample;
use stwo_prover::core::pcs::TreeVec;
use stwo_prover::core::{
    backend::{Col, Column},
    circle::{CirclePoint, Coset},
    fields::{m31::BaseField, qm31::SecureField},
    poly::{
        circle::{CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly, PolyOps},
        twiddles::TwiddleTree,
        BitReversedOrder,
    },
    ColumnVec,
};

impl PolyOps for CudaBackend {
    type Twiddles = cuda::BaseFieldVec;

    fn new_canonical_ordered(
        coset: CanonicCoset,
        values: Col<Self, BaseField>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        let size = values.len();
        let device_ptr = unsafe {
            cuda::bindings::sort_values_and_permute_with_bit_reverse_order(values.device_ptr, size)
        };
        let result = cuda::BaseFieldVec::new(device_ptr, size);
        CircleEvaluation::new(coset.circle_domain(), result)
    }

    fn interpolate(
        eval: CircleEvaluation<Self, BaseField, BitReversedOrder>,
        twiddle_tree: &TwiddleTree<Self>,
    ) -> CirclePoly<Self> {
        let values = eval.values;
        assert!(eval
            .domain
            .half_coset
            .is_doubling_of(twiddle_tree.root_coset));
        unsafe {
            cuda::bindings::interpolate(
                eval.domain.half_coset.size() as u32,
                values.device_ptr,
                twiddle_tree.itwiddles.device_ptr,
                twiddle_tree.itwiddles.len() as u32,
                values.len() as u32,
            );
        }

        CirclePoly::new(values)
    }

    fn interpolate_columns(
        columns: impl IntoIterator<Item = CircleEvaluation<Self, BaseField, BitReversedOrder>>,
        twiddles: &TwiddleTree<Self>,
    ) -> Vec<CirclePoly<Self>> {
        let columns = columns.into_iter().collect_vec();
        let values = columns
            .iter()
            .map(|column| column.values.device_ptr)
            .collect_vec();
        let number_of_rows = columns[0].len();
        unsafe {
            cuda::bindings::interpolate_columns(
                columns[0].domain.half_coset.size() as u32,
                values.as_ptr(),
                twiddles.itwiddles.device_ptr,
                twiddles.itwiddles.len() as u32,
                columns.len() as u32,
                number_of_rows as u32,
            );
        }

        columns
            .into_iter()
            .map(|column| CirclePoly::new(column.values))
            .collect_vec()
    }

    fn eval_at_point(poly: &CirclePoly<Self>, point: CirclePoint<SecureField>) -> SecureField {
        unsafe {
            cuda::bindings::eval_at_point(
                poly.coeffs.device_ptr,
                poly.coeffs.len() as u32,
                CudaSecureField::from(point.x),
                CudaSecureField::from(point.y),
            )
            .into()
        }
    }

    fn evaluate_polynomials_out_of_domain(
        polynomials: TreeVec<ColumnVec<&CirclePoly<Self>>>,
        points: TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>>,
    ) -> TreeVec<ColumnVec<Vec<PointSample>>> {
        TreeVec::new(
        polynomials
                .0
                .into_iter()
                .enumerate()
                .map(|(index, column)| {
                    let polynomial_sizes: Vec<u32> = column.iter().map( |polynomial| 1 << polynomial.log_size() ).collect();
                    let polynomial_coefficients: Vec<*const u32> = column.into_iter().map( |polynomial| polynomial.coeffs.device_ptr ).collect();
                    let out_of_domain_points = &points.0[index];
                    let points_x = out_of_domain_points.iter().map( |points_x_y|
                        points_x_y.iter().map( |point| CudaSecureField::from(point.x) ).collect_vec()
                    ).collect_vec();
                    let points_y = out_of_domain_points.iter().map( |points_x_y|
                        points_x_y.iter().map( |point| CudaSecureField::from(point.y) ).collect_vec()
                    ).collect_vec();
                    let points_x_pointers = points_x.iter().map( |point_x_for_polynomial|
                        point_x_for_polynomial.as_ptr()
                    ).collect_vec();
                    let points_y_pointers = points_y.iter().map( |point_y_for_polynomial|
                        point_y_for_polynomial.as_ptr()
                    ).collect_vec();
                    let sample_sizes = out_of_domain_points.iter().map( |points_x_y|
                        points_x_y.len() as u32
                    ).collect_vec();

                    let evaluations: Vec<Vec<CudaSecureField>> = (0..polynomial_coefficients.len())
                        .map( |index|{
                            let mut vector = Vec::with_capacity(sample_sizes[index] as usize);
                            unsafe {
                                vector.set_len(sample_sizes[index] as usize);
                            }
                            vector
                        }).collect();
                    let evaluation_pointers = evaluations
                        .iter()
                        .map(|evaluation_vector|
                            evaluation_vector.as_ptr()
                        ).collect_vec();

                    unsafe {
                        cuda::bindings::evaluate_polynomials_out_of_domain(
                            evaluation_pointers.as_ptr(),
                            polynomial_coefficients.as_ptr(),
                            polynomial_sizes.as_ptr(),
                            polynomial_coefficients.len() as u32,
                            points_x_pointers.as_ptr(),
                            points_y_pointers.as_ptr(),
                            sample_sizes.as_ptr(),
                        );
                    }

                    evaluations
                        .into_iter()
                        .zip(out_of_domain_points)
                        .map(|(values, evaluated_points)|
                            values
                                .into_iter()
                                .zip(evaluated_points)
                                .map( |(value, point)|
                                    PointSample {
                                        point: *point,
                                        value: SecureField::from(value),
                                    }
                                ).collect()
                        ).collect()
                }).collect(),
        )
    }

    fn extend(poly: &CirclePoly<Self>, log_size: u32) -> CirclePoly<Self> {
        let new_size = 1 << log_size;
        assert!(
            new_size >= poly.coeffs.len(),
            "New size must be larger than the old size"
        );

        let mut new_coeffs = cuda::BaseFieldVec::new_zeroes(new_size);
        new_coeffs.copy_from(&poly.coeffs);
        CirclePoly::new(new_coeffs)
    }

    fn evaluate(
        poly: &CirclePoly<Self>,
        domain: CircleDomain,
        twiddle_tree: &TwiddleTree<Self>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        let values = poly.extend(domain.log_size()).coeffs;
        assert!(domain.half_coset.is_doubling_of(twiddle_tree.root_coset));
        unsafe {
            cuda::bindings::evaluate(
                domain.half_coset.size() as u32,
                values.device_ptr,
                twiddle_tree.twiddles.device_ptr,
                twiddle_tree.twiddles.len() as u32,
                values.len() as u32,
            );
        }

        CircleEvaluation::new(domain, values)
    }

    fn evaluate_polynomials(
        polynomials: &mut ColumnVec<CirclePoly<Self>>,
        log_blowup_factor: u32,
        twiddles: &TwiddleTree<Self>,
    ) -> Vec<CircleEvaluation<Self, BaseField, BitReversedOrder>> {
        let mut values_pointers = Vec::new();
        let mut values_columns = Vec::new();
        let mut column_sizes = Vec::new();
        let mut domains = Vec::new();
        let mut eval_domain_sizes = Vec::new();

        for poly in polynomials.iter() {
            let domain = CanonicCoset::new(poly.log_size() + log_blowup_factor).circle_domain();
            let values = poly.extend(domain.log_size()).coeffs;

            values_pointers.push(values.device_ptr);
            column_sizes.push(values.len() as u32);
            values_columns.push(values);
            domains.push(domain);
            eval_domain_sizes.push(domain.half_coset.size() as u32);
        }

        unsafe {
            cuda::bindings::evaluate_columns(
                eval_domain_sizes.as_ptr(),
                values_pointers.as_ptr(),
                twiddles.twiddles.device_ptr,
                twiddles.twiddles.len() as u32,
                values_columns.len() as u32,
                column_sizes.as_ptr(),
            );
        }

        domains
            .into_iter()
            .zip(values_columns.into_iter())
            .map(|(domain, values)| {
                CircleEvaluation::<Self, BaseField, BitReversedOrder>::new(domain, values)
            })
            .collect_vec()
    }

    fn precompute_twiddles(coset: Coset) -> TwiddleTree<Self> {
        unsafe {
            let twiddles = cuda::BaseFieldVec::new(
                cuda::bindings::precompute_twiddles(
                    coset.initial.into(),
                    coset.step.into(),
                    coset.size(),
                ),
                coset.size(),
            );
            let itwiddles = cuda::BaseFieldVec::new_uninitialized(coset.size());
            cuda::bindings::batch_inverse_base_field(
                twiddles.device_ptr,
                itwiddles.device_ptr,
                coset.size(),
            );
            TwiddleTree {
                root_coset: coset,
                twiddles,
                itwiddles,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use stwo_prover::core::air::mask::fixed_mask_points;
    use stwo_prover::core::fields::qm31::SecureField;
    use stwo_prover::core::pcs::TreeVec;
    use stwo_prover::core::poly::circle::{
        CanonicCoset, CircleDomain, CircleEvaluation, CirclePoly, PolyOps,
    };
    use stwo_prover::core::poly::twiddles::TwiddleTree;
    use stwo_prover::core::{
        backend::{Column, CpuBackend},
        circle::{CirclePoint, CirclePointIndex, Coset, SECURE_FIELD_CIRCLE_GEN},
        fields::m31::BaseField,
        ColumnVec,
    };

    use crate::{
        backend::CudaBackend,
        cuda::{self, BaseFieldVec},
        poly::BitReversedOrder,
    };

    #[test]
    fn test_new_canonical_ordered() {
        let log_size = 23;
        let coset = CanonicCoset::new(log_size);
        let size: usize = 1 << log_size;
        let column_data = (0..size as u32).map(BaseField::from).collect::<Vec<_>>();
        let cpu_values = column_data.clone();
        let expected_result = CpuBackend::new_canonical_ordered(coset, cpu_values);

        let column = cuda::BaseFieldVec::from_vec(column_data);
        let result = CudaBackend::new_canonical_ordered(coset, column);

        assert_eq!(result.values.to_cpu(), expected_result.values);
        assert_eq!(
            result.domain.iter().collect::<Vec<_>>(),
            expected_result.domain.iter().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_precompute_twiddles() {
        let log_size = 5;

        let half_coset = CanonicCoset::new(log_size).half_coset();
        let expected_result = CpuBackend::precompute_twiddles(half_coset);
        let twiddles = CudaBackend::precompute_twiddles(half_coset);

        assert_eq!(twiddles.twiddles.to_cpu(), expected_result.twiddles);
        assert_eq!(twiddles.itwiddles.to_cpu(), expected_result.itwiddles);
        assert_eq!(
            twiddles.root_coset.iter().collect::<Vec<_>>(),
            expected_result.root_coset.iter().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_extend() {
        let log_size = 20;
        let size = 1 << log_size;
        let new_log_size = log_size + 5;
        let cpu_coeffs = (0..size).map(BaseField::from).collect::<Vec<_>>();
        let cuda_coeffs = cuda::BaseFieldVec::from_vec(cpu_coeffs.clone());
        let cpu_poly = CirclePoly::<CpuBackend>::new(cpu_coeffs);
        let cuda_poly = CirclePoly::<CudaBackend>::new(cuda_coeffs);
        let result = CudaBackend::extend(&cuda_poly, new_log_size);
        let expected_result = CpuBackend::extend(&cpu_poly, new_log_size);
        assert_eq!(result.coeffs.to_cpu(), expected_result.coeffs);
        assert_eq!(result.log_size(), expected_result.log_size());
    }

    #[test]
    fn test_interpolate() {
        let log_size = 20;

        let size = 1 << log_size;

        let cpu_values = (1..(size + 1) as u32)
            .map(BaseField::from)
            .collect::<Vec<_>>();
        let gpu_values = cuda::BaseFieldVec::from_vec(cpu_values.clone());

        let coset = CanonicCoset::new(log_size);
        let cpu_evaluations = CpuBackend::new_canonical_ordered(coset, cpu_values);
        let gpu_evaluations = CudaBackend::new_canonical_ordered(coset, gpu_values);

        let cpu_twiddles = CpuBackend::precompute_twiddles(coset.half_coset());
        let gpu_twiddles = CudaBackend::precompute_twiddles(coset.half_coset());

        let expected_result = CpuBackend::interpolate(cpu_evaluations, &cpu_twiddles);
        let result = CudaBackend::interpolate(gpu_evaluations, &gpu_twiddles);

        assert_eq!(result.coeffs.to_cpu(), expected_result.coeffs);
    }

    #[test]
    fn test_interpolate_2() {
        let log_size = 5;

        let cpu_values = vec![
            BaseField::from(1),
            BaseField::from(443693538),
            BaseField::from(793699796),
            BaseField::from(1631104375),
            BaseField::from(460025527),
            BaseField::from(98131605),
            BaseField::from(1292025643),
            BaseField::from(1056169651),
            BaseField::from(29),
            BaseField::from(1645907698),
            BaseField::from(300234932),
            BaseField::from(2113642380),
            BaseField::from(2031046861),
            BaseField::from(541052612),
            BaseField::from(1857203558),
            BaseField::from(5),
            BaseField::from(2),
            BaseField::from(187770177),
            BaseField::from(1190378570),
            BaseField::from(1107054997),
            BaseField::from(1436440899),
            BaseField::from(1555024221),
            BaseField::from(2002021885),
            BaseField::from(866),
            BaseField::from(750797),
            BaseField::from(1704111751),
            BaseField::from(1874758341),
            BaseField::from(960394553),
            BaseField::from(1365348280),
            BaseField::from(376645196),
            BaseField::from(2119137245),
            BaseField::from(1),
        ];
        let gpu_values = cuda::BaseFieldVec::from_vec(cpu_values.clone());

        let coset = CanonicCoset::new(log_size);
        let cpu_evaluations = CpuBackend::new_canonical_ordered(coset, cpu_values);
        let gpu_evaluations = CudaBackend::new_canonical_ordered(coset, gpu_values);

        let cpu_twiddles = CpuBackend::precompute_twiddles(coset.half_coset());
        // println!("{:?}", cpu_twiddles.twiddles);
        let gpu_twiddles = CudaBackend::precompute_twiddles(coset.half_coset());

        let expected_result = CpuBackend::interpolate(cpu_evaluations, &cpu_twiddles);
        let result = CudaBackend::interpolate(gpu_evaluations, &gpu_twiddles);

        assert_eq!(result.coeffs.to_cpu(), expected_result.coeffs);
    }

    #[test]
    fn test_evaluate() {
        let log_size = 20;

        let size = 1 << log_size;

        let cpu_values = (1..(size + 1) as u32)
            .map(BaseField::from)
            .collect::<Vec<_>>();
        let gpu_values = cuda::BaseFieldVec::from_vec(cpu_values.clone());

        let coset = CanonicCoset::new(log_size);
        let cpu_evaluations = CpuBackend::new_canonical_ordered(coset, cpu_values);
        let gpu_evaluations = CudaBackend::new_canonical_ordered(coset, gpu_values);

        let cpu_twiddles = CpuBackend::precompute_twiddles(coset.half_coset());
        let gpu_twiddles = CudaBackend::precompute_twiddles(coset.half_coset());

        let cpu_poly = CpuBackend::interpolate(cpu_evaluations, &cpu_twiddles);
        let gpu_poly = CudaBackend::interpolate(gpu_evaluations, &gpu_twiddles);

        let expected_result = CpuBackend::evaluate(&cpu_poly, coset.circle_domain(), &cpu_twiddles);
        let result = CudaBackend::evaluate(&gpu_poly, coset.circle_domain(), &gpu_twiddles);

        assert_eq!(result.values.to_cpu(), expected_result.values);
    }

    #[test]
    fn test_eval_at_point() {
        let log_size = 20;

        let size = 1 << log_size;
        let coset = CanonicCoset::new(log_size);
        let point = SECURE_FIELD_CIRCLE_GEN;

        let cpu_values = (1..(size + 1) as u32)
            .map(BaseField::from)
            .collect::<Vec<_>>();

        let gpu_values = cuda::BaseFieldVec::from_vec(cpu_values.clone());
        let gpu_evaluations = CudaBackend::new_canonical_ordered(coset, gpu_values);
        let gpu_twiddles = CudaBackend::precompute_twiddles(coset.half_coset());
        let gpu_poly = CudaBackend::interpolate(gpu_evaluations, &gpu_twiddles);
        let result = CudaBackend::eval_at_point(&gpu_poly, point);

        let cpu_evaluations = CpuBackend::new_canonical_ordered(coset, cpu_values);
        let cpu_twiddles = CpuBackend::precompute_twiddles(coset.half_coset());
        let cpu_poly = CpuBackend::interpolate(cpu_evaluations, &cpu_twiddles);

        let expected_result = CpuBackend::eval_at_point(&cpu_poly, point.clone());

        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_interpolate_from_fib() {
        let eval = CircleEvaluation::<CudaBackend, BaseField, BitReversedOrder>::new(
            CircleDomain {
                half_coset: Coset {
                    initial_index: CirclePointIndex(33554432),
                    initial: CirclePoint {
                        x: BaseField::from(579625837),
                        y: BaseField::from(1690787918),
                    },
                    step_size: CirclePointIndex(134217728),
                    step: CirclePoint {
                        x: BaseField::from(590768354),
                        y: BaseField::from(978592373),
                    },
                    log_size: 4,
                },
            },
            BaseFieldVec::from_vec(vec![
                BaseField::from(1),
                BaseField::from(443693538),
                BaseField::from(793699796),
                BaseField::from(1631104375),
                BaseField::from(460025527),
                BaseField::from(98131605),
                BaseField::from(1292025643),
                BaseField::from(1056169651),
                BaseField::from(29),
                BaseField::from(1645907698),
                BaseField::from(300234932),
                BaseField::from(2113642380),
                BaseField::from(2031046861),
                BaseField::from(541052612),
                BaseField::from(1857203558),
                BaseField::from(5),
                BaseField::from(2),
                BaseField::from(187770177),
                BaseField::from(1190378570),
                BaseField::from(1107054997),
                BaseField::from(1436440899),
                BaseField::from(1555024221),
                BaseField::from(2002021885),
                BaseField::from(866),
                BaseField::from(750797),
                BaseField::from(1704111751),
                BaseField::from(1874758341),
                BaseField::from(960394553),
                BaseField::from(1365348280),
                BaseField::from(376645196),
                BaseField::from(2119137245),
                BaseField::from(1),
            ]),
        );
        let twiddles = vec![
            BaseField::from(785043271),
            BaseField::from(1260750973),
            BaseField::from(736262640),
            BaseField::from(1553669210),
            BaseField::from(479120236),
            BaseField::from(225856549),
            BaseField::from(197700101),
            BaseField::from(1079800039),
            BaseField::from(1911378744),
            BaseField::from(1577470940),
            BaseField::from(1334497267),
            BaseField::from(2085743640),
            BaseField::from(477953613),
            BaseField::from(125103457),
            BaseField::from(1977033713),
            BaseField::from(2005527287),
            BaseField::from(251924953),
            BaseField::from(636875771),
            BaseField::from(48903418),
            BaseField::from(1896945393),
            BaseField::from(1514613395),
            BaseField::from(870936612),
            BaseField::from(1297878576),
            BaseField::from(583555490),
            BaseField::from(640817200),
            BaseField::from(1702126977),
            BaseField::from(1054411686),
            BaseField::from(648593218),
            BaseField::from(1014093253),
            BaseField::from(2137011181),
            BaseField::from(81378258),
            BaseField::from(789857006),
            BaseField::from(838195206),
            BaseField::from(1774253895),
            BaseField::from(1739004854),
            BaseField::from(262191051),
            BaseField::from(206059115),
            BaseField::from(212443077),
            BaseField::from(1796741361),
            BaseField::from(883753057),
            BaseField::from(2140339328),
            BaseField::from(404685994),
            BaseField::from(9803698),
            BaseField::from(68458636),
            BaseField::from(14530030),
            BaseField::from(228509164),
            BaseField::from(1038945916),
            BaseField::from(134155457),
            BaseField::from(579625837),
            BaseField::from(1690787918),
            BaseField::from(1641940819),
            BaseField::from(2121318970),
            BaseField::from(1952787376),
            BaseField::from(1580223790),
            BaseField::from(1013961365),
            BaseField::from(280947147),
            BaseField::from(1179735656),
            BaseField::from(1241207368),
            BaseField::from(1415090252),
            BaseField::from(2112881577),
            BaseField::from(590768354),
            BaseField::from(978592373),
            BaseField::from(32768),
            BaseField::from(1),
        ];
        let itwiddles = vec![
            BaseField::from(1541158724),
            BaseField::from(16208603),
            BaseField::from(62823040),
            BaseField::from(1642210396),
            BaseField::from(1631996251),
            BaseField::from(1007591000),
            BaseField::from(1874949287),
            BaseField::from(1849862501),
            BaseField::from(781334166),
            BaseField::from(132945364),
            BaseField::from(1278220752),
            BaseField::from(214347122),
            BaseField::from(1165838173),
            BaseField::from(2054194025),
            BaseField::from(1234096940),
            BaseField::from(1721693449),
            BaseField::from(622651690),
            BaseField::from(1373671071),
            BaseField::from(82740187),
            BaseField::from(1683898894),
            BaseField::from(1918467639),
            BaseField::from(1186332607),
            BaseField::from(1296073347),
            BaseField::from(401388709),
            BaseField::from(1383565722),
            BaseField::from(656788371),
            BaseField::from(1787268380),
            BaseField::from(1809670981),
            BaseField::from(99372120),
            BaseField::from(765975505),
            BaseField::from(774809712),
            BaseField::from(348924564),
            BaseField::from(2029303208),
            BaseField::from(959596234),
            BaseField::from(1051468699),
            BaseField::from(721860568),
            BaseField::from(1767118503),
            BaseField::from(218253990),
            BaseField::from(1356867335),
            BaseField::from(1955048591),
            BaseField::from(559361447),
            BaseField::from(1046725194),
            BaseField::from(448375059),
            BaseField::from(1036402186),
            BaseField::from(2138687850),
            BaseField::from(1268642696),
            BaseField::from(1381082522),
            BaseField::from(559888787),
            BaseField::from(248349974),
            BaseField::from(969924856),
            BaseField::from(1461702947),
            BaseField::from(655012266),
            BaseField::from(1385854532),
            BaseField::from(1859156789),
            BaseField::from(349252128),
            BaseField::from(421110815),
            BaseField::from(1160411471),
            BaseField::from(1518526074),
            BaseField::from(490549293),
            BaseField::from(1942501404),
            BaseField::from(991237807),
            BaseField::from(775648038),
            BaseField::from(65536),
            BaseField::from(1),
        ];
        let root_coset = Coset {
            initial_index: CirclePointIndex(8388608),
            initial: CirclePoint {
                x: BaseField::from(785043271),
                y: BaseField::from(1260750973),
            },
            step_size: CirclePointIndex(33554432),
            step: CirclePoint {
                x: BaseField::from(579625837),
                y: BaseField::from(1690787918),
            },
            log_size: 6,
        };
        let twiddle_tree = TwiddleTree::<CudaBackend> {
            root_coset,
            twiddles: BaseFieldVec::from_vec(twiddles),
            itwiddles: BaseFieldVec::from_vec(itwiddles),
        };

        let cpu_evaluation = CircleEvaluation::<CpuBackend, BaseField, BitReversedOrder>::new(
            eval.domain,
            eval.values.to_cpu(),
        );
        let cpu_twiddle_tree = TwiddleTree::<CpuBackend> {
            root_coset: twiddle_tree.root_coset.clone(),
            twiddles: twiddle_tree.twiddles.to_cpu(),
            itwiddles: twiddle_tree.itwiddles.to_cpu(),
        };
        let expected_result = CpuBackend::interpolate(cpu_evaluation, &cpu_twiddle_tree);
        let result = CudaBackend::interpolate(eval, &twiddle_tree);
        assert_eq!(expected_result.coeffs, result.coeffs.to_cpu());
    }

    #[test]
    fn test_interpolate_columns() {
        let log_size = 9;
        let log_number_of_columns = 8;

        let size = 1 << log_size;
        let number_of_columns = 1 << log_number_of_columns;

        let cpu_values = (1..(size + 1) as u32).map(BaseField::from).collect_vec();
        let gpu_values = cuda::BaseFieldVec::from_vec(cpu_values.clone());

        let coset = CanonicCoset::new(log_size);
        let cpu_evaluations = CpuBackend::new_canonical_ordered(coset, cpu_values);
        let gpu_evaluations = CudaBackend::new_canonical_ordered(coset, gpu_values);

        let cpu_twiddles = CpuBackend::precompute_twiddles(coset.half_coset());
        let gpu_twiddles = CudaBackend::precompute_twiddles(coset.half_coset());

        let cpu_columns = (0..number_of_columns)
            .map(|_index| cpu_evaluations.clone())
            .collect_vec();
        let gpu_columns = (0..number_of_columns)
            .map(|_index| gpu_evaluations.clone())
            .collect_vec();

        let expected_result = CpuBackend::interpolate_columns(cpu_columns, &cpu_twiddles);
        let result = CudaBackend::interpolate_columns(gpu_columns, &gpu_twiddles);

        let expected_coeffs = expected_result
            .iter()
            .map(|poly| poly.coeffs.clone())
            .collect_vec();
        let coeffs = result
            .iter()
            .map(|poly| poly.coeffs.clone().to_cpu())
            .collect_vec();

        assert_eq!(coeffs, expected_coeffs);
    }

    #[test]
    fn test_evaluate_polynomials() {
        let log_size = 9;
        let log_number_of_columns = 8;
        let log_blowup_factor = 2;

        let size = 1 << log_size;
        let number_of_columns = 1 << log_number_of_columns;

        let cpu_values = (1..(size + 1) as u32)
            .map(BaseField::from)
            .collect::<Vec<_>>();
        let gpu_values = cuda::BaseFieldVec::from_vec(cpu_values.clone());

        let trace_coset = CanonicCoset::new(log_size);
        let cpu_evaluations = CpuBackend::new_canonical_ordered(trace_coset, cpu_values);
        let gpu_evaluations = CudaBackend::new_canonical_ordered(trace_coset, gpu_values);

        let interpolation_coset = CanonicCoset::new(log_size + log_blowup_factor);
        let cpu_twiddles = CpuBackend::precompute_twiddles(interpolation_coset.half_coset());
        let gpu_twiddles = CudaBackend::precompute_twiddles(interpolation_coset.half_coset());

        let cpu_poly = CpuBackend::interpolate(cpu_evaluations, &cpu_twiddles);
        let gpu_poly = CudaBackend::interpolate(gpu_evaluations, &gpu_twiddles);

        let mut cpu_columns = ColumnVec::from(
            (0..number_of_columns)
                .map(|_index| cpu_poly.clone())
                .collect_vec(),
        );
        let mut gpu_columns = ColumnVec::from(
            (0..number_of_columns)
                .map(|_index| gpu_poly.clone())
                .collect_vec(),
        );

        let result = CudaBackend::evaluate_polynomials(&mut gpu_columns, log_blowup_factor, &gpu_twiddles);
        let expected_result =
            CpuBackend::evaluate_polynomials(&mut cpu_columns, log_blowup_factor, &cpu_twiddles);

        let expected_values = expected_result
            .iter()
            .map(|eval| eval.clone().values)
            .collect_vec();
        let values = result
            .iter()
            .map(|eval| eval.clone().values.to_cpu())
            .collect_vec();

        assert_eq!(values, expected_values);
    }

    fn generate_random_point() -> CirclePoint<SecureField> {
        let mut rng = SmallRng::seed_from_u64(0);
        let x = rng.gen();
        let y = rng.gen();
        CirclePoint { x, y }
    }
    
    fn mask_points(
        point: CirclePoint<SecureField>,
        number_of_columns: usize,
    ) -> TreeVec<ColumnVec<Vec<CirclePoint<SecureField>>>> {
        TreeVec(vec![fixed_mask_points(
            &vec![vec![0_usize]; number_of_columns],
            point,
        )])
    }

    #[test]
    fn test_evaluate_polynomials_out_of_domain() {
        let log_size = 9;
        let log_number_of_columns = 8;
        let log_blowup_factor = 2;

        let size = 1 << log_size;
        let number_of_columns = 1 << log_number_of_columns;

        let cpu_values = (1..(size + 1) as u32)
            .map(BaseField::from)
            .collect::<Vec<_>>();
        let gpu_values = cuda::BaseFieldVec::from_vec(cpu_values.clone());

        let trace_coset = CanonicCoset::new(log_size);
        let cpu_evaluations = CpuBackend::new_canonical_ordered(trace_coset, cpu_values);
        let gpu_evaluations = CudaBackend::new_canonical_ordered(trace_coset, gpu_values);

        let interpolation_coset = CanonicCoset::new(log_size + log_blowup_factor);
        let cpu_twiddles = CpuBackend::precompute_twiddles(interpolation_coset.half_coset());
        let gpu_twiddles = CudaBackend::precompute_twiddles(interpolation_coset.half_coset());

        let cpu_poly = CpuBackend::interpolate(cpu_evaluations, &cpu_twiddles);
        let gpu_poly = CudaBackend::interpolate(gpu_evaluations, &gpu_twiddles);

        let cpu_polynomials = ColumnVec::from(
            (0..number_of_columns)
                .map(|_index| &cpu_poly)
                .collect_vec(),
        );
        let gpu_polynomials = ColumnVec::from(
            (0..number_of_columns)
                .map(|_index| &gpu_poly)
                .collect_vec(),
        );

        let point = generate_random_point();
        let sample_points = mask_points(point, number_of_columns);

        let result = CudaBackend::evaluate_polynomials_out_of_domain(TreeVec::new(vec![gpu_polynomials]), sample_points.clone());
        let expected_result =
            CpuBackend::evaluate_polynomials_out_of_domain(TreeVec::new(vec![cpu_polynomials]), sample_points.clone());

        let flattened_result = result.flatten_cols();
        let flattened_expected_result = expected_result.flatten_cols();

        let values = flattened_result
            .iter()
            .map(|point_sample| (point_sample.point, point_sample.value))
            .collect_vec();
        let expected_values = flattened_expected_result
            .iter()
            .map(|point_sample| (point_sample.point, point_sample.value))
            .collect_vec();

        assert_eq!(values, expected_values);
    }
}
