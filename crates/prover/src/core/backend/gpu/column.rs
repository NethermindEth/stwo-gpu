use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice, LaunchAsync};
use cudarc::nvrtc::compile_ptx;
use itertools::izip;
use itertools::Itertools;
use num_traits::Zero;

use super::{GpuBackend, DEVICE};
use crate::core::backend::{Column, CpuBackend};
use crate::core::fields::m31::{BaseField, M31};
use crate::core::fields::qm31::{SecureField, QM31};
use crate::core::fields::secure_column::SecureColumn;
use crate::core::fields::FieldOps;

impl FieldOps<BaseField> for GpuBackend {
    fn batch_inverse(from: &Self::Column, dst: &mut Self::Column) {
        let size = from.len();
        let log_size = u32::BITS - (size as u32).leading_zeros() - 1;

        // Shared memory:
        // 512 bytes to store the `from` chunk in shared memory.
        // 512 - 32 bytes to store the inner tree up to the level with 32 elements.
        let config = Self::launch_config_for_num_elems(size as u32 >> 1, 256, 512 * 4  + (512 - 32) * 4);
        let batch_inverse = DEVICE
            .get_func("column", "batch_inverse_basefield")
            .unwrap();
        unsafe {
            let res = batch_inverse.launch(
                config,
                (from.as_slice(), dst.as_mut_slice(), size, log_size),
            );
            res
        }
        .unwrap();
        DEVICE.synchronize().unwrap();
    }
}

impl FieldOps<SecureField> for GpuBackend {
    fn batch_inverse(from: &Self::Column, dst: &mut Self::Column) {
        let size = from.len();
        let log_size = u32::BITS - (size as u32).leading_zeros() - 1;

        // Shared memory:
        // 1024 * 4 bytes to store the `from` chunk in shared memory.
        // (1024 - 32) * 4 bytes to store the inner tree up to the level with 32 elements.
        let config = Self::launch_config_for_num_elems(size as u32 >> 1, 512, 1024 * 4 * 4 + (1024 - 32) * 4 * 4);
        let batch_inverse = DEVICE
            .get_func("column", "batch_inverse_secure_field")
            .unwrap();
        unsafe {
            let res = batch_inverse.launch(
                config,
                (from.as_slice(), dst.as_mut_slice(), size, log_size),
            );
            res
        }
        .unwrap();
        DEVICE.synchronize().unwrap();
    }
}

#[derive(Debug, Clone)]
pub struct BaseFieldCudaColumn(CudaSlice<M31>);

#[allow(unused)]
impl BaseFieldCudaColumn {
    pub fn new(column: CudaSlice<M31>) -> Self {
        Self(column)
    }

    pub fn from_vec(column: Vec<M31>) -> Self {
        Self(DEVICE.htod_sync_copy(&column).unwrap())
    }

    pub fn as_mut_slice(&mut self) -> &mut CudaSlice<M31> {
        &mut self.0
    }

    pub fn as_slice(&self) -> &CudaSlice<M31> {
        &self.0
    }
}

impl FromIterator<BaseField> for BaseFieldCudaColumn {
    fn from_iter<T: IntoIterator<Item = BaseField>>(iter: T) -> Self {
        BaseFieldCudaColumn::new(DEVICE.htod_copy(iter.into_iter().collect_vec()).unwrap())
    }
}

impl Column<BaseField> for BaseFieldCudaColumn {
    fn zeros(len: usize) -> Self {
        // TODO: optimize
        let zeroes_vec = vec![BaseField::zero(); len];
        Self::from_vec(zeroes_vec)
    }

    fn to_cpu(&self) -> Vec<BaseField> {
        DEVICE.dtoh_sync_copy(&self.0).unwrap()
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn at(&self, _index: usize) -> BaseField {
        todo!()
    }
}

#[derive(Debug, Clone)]
pub struct SecureFieldCudaColumn(CudaSlice<QM31>);

#[allow(unused)]
impl SecureFieldCudaColumn {
    pub fn new(column: CudaSlice<QM31>) -> Self {
        Self(column)
    }

    pub fn from_vec(column: Vec<QM31>) -> Self {
        Self(DEVICE.htod_sync_copy(&column).unwrap())
    }

    pub fn as_mut_slice(&mut self) -> &mut CudaSlice<QM31> {
        &mut self.0
    }

    pub fn as_slice(&self) -> &CudaSlice<QM31> {
        &self.0
    }
}

impl FromIterator<SecureField> for SecureFieldCudaColumn {
    fn from_iter<T: IntoIterator<Item = SecureField>>(iter: T) -> Self {
        Self(DEVICE.htod_sync_copy(&iter.into_iter().collect::<Vec<QM31>>()).unwrap())
    }
}

impl FromIterator<SecureField> for SecureColumn<GpuBackend> {
    fn from_iter<I: IntoIterator<Item = SecureField>>(iter: I) -> Self {
        let cpu_col = SecureColumn::<CpuBackend>::from_iter(iter);
        let columns = cpu_col.columns.map(|col| col.into_iter().collect());
        SecureColumn { columns }
    }
}


impl SecureColumn<GpuBackend> {
    pub fn to_vec(&self) -> Vec<SecureField> {
        izip!(
            self.columns[0].to_cpu(),
            self.columns[1].to_cpu(),
            self.columns[2].to_cpu(),
            self.columns[3].to_cpu(),
        )
        .map(|(a, b, c, d)| SecureField::from_m31_array([a, b, c, d]))
        .collect()
    }
}




impl Column<SecureField> for SecureFieldCudaColumn {
    fn zeros(_len: usize) -> Self {
        todo!()
    }

    fn to_cpu(&self) -> Vec<SecureField> {
        DEVICE.dtoh_sync_copy(&self.0).unwrap()
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn at(&self, _index: usize) -> SecureField {
        todo!()
    }
}

pub fn load_batch_inverse_ptx(device: &Arc<CudaDevice>) {
    let ptx_src = include_str!("column.cu");
    let ptx = compile_ptx(ptx_src).unwrap();
    device
        .load_ptx(
            ptx,
            "column",
            &["batch_inverse_basefield", "batch_inverse_secure_field"],
        )
        .unwrap();
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use crate::core::backend::gpu::column::{BaseFieldCudaColumn, SecureFieldCudaColumn};
    use crate::core::backend::gpu::GpuBackend;
    use crate::core::backend::{Column, CpuBackend};
    use crate::core::fields::m31::{BaseField, M31};
    use crate::core::fields::qm31::{SecureField, QM31};
    use crate::core::fields::FieldOps;

    #[test]
    fn test_batch_inverse_basefield() {
        let size: usize = 1 << 24;
        let from = (1..(size + 1) as u32).map(|x| M31(x)).collect_vec();
        let dst = from.clone();
        let mut dst_expected = dst.clone();
        CpuBackend::batch_inverse(&from, &mut dst_expected);

        let from_device = BaseFieldCudaColumn::from_vec(from);
        let mut dst_device = BaseFieldCudaColumn::from_vec(dst);
        <GpuBackend as FieldOps<BaseField>>::batch_inverse(&from_device, &mut dst_device);

        assert_eq!(dst_device.to_cpu(), dst_expected.to_cpu());
    }

    #[test]
    fn test_batch_inverse_secure_field() {
        let size: usize = 1 << 25;

        let from_raw = (1..(size + 1) as u32).collect::<Vec<u32>>();

        let from_cpu = from_raw
            .chunks(4)
            .map(|a| QM31::from_u32_unchecked(a[0], a[1], a[2], a[3]))
            .collect_vec();
        let mut dst_expected_cpu = from_cpu.clone();

        CpuBackend::batch_inverse(&from_cpu, &mut dst_expected_cpu);

        let from_device = SecureFieldCudaColumn::from_vec(from_cpu.clone());
        let mut dst_device = SecureFieldCudaColumn::from_vec(from_cpu.clone());

        <GpuBackend as FieldOps<SecureField>>::batch_inverse(&from_device, &mut dst_device);

        assert_eq!(dst_device.to_cpu(), dst_expected_cpu);
    }
}
