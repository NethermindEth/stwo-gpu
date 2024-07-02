use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

use super::column::{BaseFieldCudaColumn};
use super::{GpuBackend, DEVICE};
use crate::core::backend::Column;
use crate::core::fields::m31::M31;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::secure_column::SecureColumn;
use crate::core::fri::FriOps;
use crate::core::poly::circle::SecureEvaluation;
use crate::core::poly::line::LineEvaluation;
use crate::core::poly::twiddles::TwiddleTree;

impl FriOps for GpuBackend {
    fn fold_line(
        _eval: &LineEvaluation<Self>,
        _alpha: SecureField,
        _twiddles: &TwiddleTree<Self>,
    ) -> LineEvaluation<Self> {
        todo!()
    }

    fn fold_circle_into_line(
        _dst: &mut LineEvaluation<GpuBackend>,
        _src: &SecureEvaluation<Self>,
        _alpha: SecureField,
        _twiddles: &TwiddleTree<Self>,
    ) {
        todo!()
    }

    fn decompose(eval: &SecureEvaluation<Self>) -> (SecureEvaluation<Self>, SecureField) {
        let columns = &eval.columns;

        let lambda = unsafe {
           let a = Self::sum(&columns[0]);
           let b = Self::sum(&columns[1]);
           let c = Self::sum(&columns[2]);
           let d = Self::sum(&columns[3]);
            SecureField::from_m31(a, b, c, d) / M31::from_u32_unchecked(eval.len() as u32)
        };

        let g_values = unsafe {
            SecureColumn {
                columns: [
                    Self::compute_g_values(&columns[0], lambda.0.0),
                    Self::compute_g_values(&columns[1], lambda.0.1),
                    Self::compute_g_values(&columns[2], lambda.1.0),
                    Self::compute_g_values(&columns[3], lambda.1.1),
                ]
            }
        };

        let g = SecureEvaluation {
            domain: eval.domain,
            values: g_values,
        };
        (g, lambda)
    }
}

impl GpuBackend {
    unsafe fn sum(column: &BaseFieldCudaColumn) -> M31 {
        let size = column.len();
        let sum_launch_config = LaunchConfig::for_num_elems(size as u32 >> 1);
        let amount_of_partial_results = sum_launch_config.grid_dim.0;

        let mut partial_results: CudaSlice<M31> = DEVICE.alloc(amount_of_partial_results as usize).unwrap();
        let mut temp: CudaSlice<M31> = DEVICE.alloc(size >> 1).unwrap();

        let kernel = DEVICE.get_func("fri", "sum").unwrap();
        kernel.launch(
            sum_launch_config, 
            (
                column.as_slice(),
                &mut temp,
                &mut partial_results,
                size
            )
        ).unwrap();
        DEVICE.synchronize().unwrap();
        
        let mut gpu_result: CudaSlice<M31> = DEVICE.alloc(1).unwrap();
        if amount_of_partial_results == 1 {
            return DEVICE.dtoh_sync_copy(&partial_results).unwrap()[0];
        }

        // Ojo que esto puede tener más de 2 ^ 11 elementos. Testear ese caso con datos de, por ejemplo, log_size = 23.
        // let pairwise_sum_launch_config = LaunchConfig::for_num_elems(amount_of_partial_results);
        let pairwise_sum_launch_config = LaunchConfig::for_num_elems(1);
        let kernel = DEVICE.get_func("fri", "pairwise_sum").unwrap();
        kernel.launch(
            pairwise_sum_launch_config,
            (
                &mut partial_results,
                &mut temp,
                &mut gpu_result,
                amount_of_partial_results
            )
        ).unwrap();
        DEVICE.synchronize().unwrap();

        DEVICE.dtoh_sync_copy(&gpu_result).unwrap()[0]
    }

    unsafe fn compute_g_values(f_values: &BaseFieldCudaColumn, lambda: M31) -> BaseFieldCudaColumn {
        let size = f_values.len();
        let mut results: CudaSlice<M31> = DEVICE.alloc(size).unwrap();

        let kernel = DEVICE.get_func("fri", "compute_g_values").unwrap();
        kernel.launch(
            LaunchConfig::for_num_elems(size as u32), 
            (
                f_values.as_slice(), 
                &mut results, 
                lambda,
                size
            )
        ).unwrap();
        BaseFieldCudaColumn::new(results)
    }
}

pub fn load_fri(device: &Arc<CudaDevice>) {
    let ptx_src = include_str!("fri.cu");
    let ptx = compile_ptx(ptx_src).unwrap();
    device
        .load_ptx(
            ptx,
            "fri",
            &[
                "sum",
                "pairwise_sum",
                "compute_g_values"
            ],
        )
        .unwrap();
}

#[cfg(test)]
mod tests{
    use itertools::Itertools;
    use crate::core::{backend::{gpu::GpuBackend, Column, CpuBackend}, fields::qm31::QM31, fri::FriOps, poly::circle::{CanonicCoset, SecureEvaluation}};

    #[test]
    fn test_decompose() {
        let domain_log_size = 22;
        let size = 1 << domain_log_size;
        let coset = CanonicCoset::new(domain_log_size);
        let domain = coset.circle_domain();

        let from_raw = (0..size * 4 as u32).collect::<Vec<u32>>();

        let from = from_raw
            .chunks(4)
            .map(|a| QM31::from_u32_unchecked(a[0], a[1], a[2], a[3]))
            .collect_vec();

        let cpu_secure_evaluation = SecureEvaluation {
            domain: domain,
            values: from.clone().into_iter().collect()
        };

        let gpu_secure_evaluation = SecureEvaluation {
            domain: domain,
            values: from.iter().copied().collect()
        };

        let (expected_g_values, expected_lambda) = CpuBackend::decompose(&cpu_secure_evaluation);
        let (g_values, lambda) = GpuBackend::decompose(&gpu_secure_evaluation);

        assert_eq!(lambda, expected_lambda);
        assert_eq!(g_values.values.columns[0].to_cpu(), expected_g_values.values.columns[0]);
        assert_eq!(g_values.values.columns[1].to_cpu(), expected_g_values.values.columns[1]);
        assert_eq!(g_values.values.columns[2].to_cpu(), expected_g_values.values.columns[2]);
        assert_eq!(g_values.values.columns[3].to_cpu(), expected_g_values.values.columns[3]);
    }
}
