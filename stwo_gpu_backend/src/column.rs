use stwo_prover::core::{
    backend::{Column, ColumnOps},
    fields::{m31::BaseField, qm31::SecureField},
};

use crate::{backend::CudaBackend, cuda};

impl ColumnOps<BaseField> for CudaBackend {
    type Column = cuda::BaseFieldVec;

    fn bit_reverse_column(column: &mut Self::Column) {
        let size = column.len();
        assert!(size.is_power_of_two() && size < u32::MAX as usize);

        unsafe {
            cuda::bindings::bit_reverse_base_field(column.device_ptr as *const u32, size);
        }
    }
}

impl ColumnOps<SecureField> for CudaBackend {
    type Column = cuda::SecureFieldVec;

    fn bit_reverse_column(column: &mut Self::Column) {
        let size = column.len();
        assert!(size.is_power_of_two() && size < u32::MAX as usize);

        unsafe {
            cuda::bindings::bit_reverse_secure_field(column.device_ptr as *const u32, size);
        }
    }
}

impl Column<BaseField> for cuda::BaseFieldVec {
    fn zeros(len: usize) -> Self {
        todo!()
    }

    fn to_cpu(&self) -> Vec<BaseField> {
        self.to_vec()
    }

    fn len(&self) -> usize {
        self.size
    }

    fn at(&self, index: usize) -> BaseField {
        todo!()
    }

    fn set(&mut self, index: usize, value: BaseField) {
        todo!()
    }
}

impl FromIterator<BaseField> for cuda::BaseFieldVec {
    fn from_iter<T: IntoIterator<Item = BaseField>>(iter: T) -> Self {
        todo!()
    }
}

impl Column<SecureField> for cuda::SecureFieldVec {
    fn zeros(len: usize) -> Self {
        todo!()
    }

    fn to_cpu(&self) -> Vec<SecureField> {
        self.to_vec()
    }

    fn len(&self) -> usize {
        self.size
    }

    fn at(&self, index: usize) -> SecureField {
        todo!()
    }

    fn set(&mut self, index: usize, value: SecureField) {
        todo!()
    }
}

impl FromIterator<SecureField> for cuda::SecureFieldVec {
    fn from_iter<T: IntoIterator<Item = SecureField>>(iter: T) -> Self {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use stwo_prover::core::{
        backend::{Column, ColumnOps, CpuBackend},
        fields::{m31::BaseField, qm31::SecureField},
    };

    use crate::{
        backend::CudaBackend,
        cuda::{BaseFieldVec, SecureFieldVec},
    };

    #[test]
    fn test_bit_reverse_base_field() {
        let size: usize = 1 << 12;
        let column_data = (0..size as u32).map(BaseField::from).collect::<Vec<_>>();
        let mut expected_result = column_data.clone();
        CpuBackend::bit_reverse_column(&mut expected_result);

        let mut column = BaseFieldVec::new(column_data);
        <CudaBackend as ColumnOps<BaseField>>::bit_reverse_column(&mut column);

        assert_eq!(column.to_cpu(), expected_result);
    }

    #[test]
    fn test_bit_reverse_secure_field() {
        let size: usize = 1 << 12;

        let from_raw = (1..(size + 1) as u32).collect::<Vec<u32>>();
        let from_cpu = from_raw
            .chunks(4)
            .map(|a| SecureField::from_u32_unchecked(a[0], a[1], a[2], a[3]))
            .collect::<Vec<_>>();
        let mut array_expected = from_cpu.clone();

        CpuBackend::bit_reverse_column(&mut array_expected);

        let mut array = SecureFieldVec::new(from_cpu.clone());
        <CudaBackend as ColumnOps<SecureField>>::bit_reverse_column(&mut array);

        assert_eq!(array.to_cpu(), array_expected);
    }
}
