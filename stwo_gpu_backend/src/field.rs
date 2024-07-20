use stwo_prover::core::{
    backend::Column,
    fields::{m31::BaseField, qm31::SecureField, FieldOps},
};

use crate::{backend::CudaBackend, cuda};

impl FieldOps<BaseField> for CudaBackend {
    fn batch_inverse(column: &Self::Column, dst: &mut Self::Column) {
        let size = column.len();
        let bits = u32::BITS - (size as u32).leading_zeros() - 1;
        unsafe {
            cuda::bindings::batch_inverse_base_field(
                column.device_ptr,
                dst.device_ptr,
                size,
                bits as usize,
            );
        }
    }
}

impl FieldOps<SecureField> for CudaBackend {
    fn batch_inverse(column: &Self::Column, dst: &mut Self::Column) {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use stwo_prover::core::{
        backend::{Column, CpuBackend},
        fields::{m31::BaseField, FieldOps},
    };

    use crate::{backend::CudaBackend, cuda};

    #[test]
    fn test_batch_inverse_basefield() {
        let size: usize = 1 << 25;
        let from = (1..(size + 1) as u32)
            .map(|x| BaseField::from(x))
            .collect::<Vec<_>>();
        let dst = from.clone();
        let mut dst_expected = dst.clone();
        CpuBackend::batch_inverse(&from, &mut dst_expected);

        let from_device = cuda::BaseFieldVec::new(from);
        let mut dst_device = cuda::BaseFieldVec::new(dst);
        <CudaBackend as FieldOps<BaseField>>::batch_inverse(&from_device, &mut dst_device);

        assert_eq!(dst_device.to_cpu(), dst_expected.to_cpu());
    }
}
