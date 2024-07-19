use stwo_prover::core::{backend::Column, fields::m31::BaseField};

use crate::cuda::bindings;

#[derive(Clone, Debug)]
pub struct BaseFieldCudaColumn {
    pub(crate) device_ptr: *const u32,
    pub(crate) size: usize,
}

impl Column<BaseField> for BaseFieldCudaColumn {
    fn zeros(len: usize) -> Self {
        todo!()
    }

    fn to_cpu(&self) -> Vec<BaseField> {
        base_field_cuda_column_to_vec(&self)
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

impl FromIterator<BaseField> for BaseFieldCudaColumn {
    fn from_iter<T: IntoIterator<Item = BaseField>>(iter: T) -> Self {
        todo!()
    }
}

pub fn base_field_cuda_column_to_vec(column: &BaseFieldCudaColumn) -> Vec<BaseField> {
    let mut host_data: Vec<BaseField> = Vec::with_capacity(column.size);
    unsafe {
        host_data.set_len(column.size.try_into().unwrap());
        bindings::m31_device_to_host(
            column.device_ptr,
            host_data.as_mut_ptr() as *const u32,
            column.size as u32,
        );
    }
    host_data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1() {
        let size = 1 << 3;

        let cuda_column = BaseFieldCudaColumn {
            device_ptr: unsafe { bindings::generate_array(size) },
            size: size as usize,
        };

        let value = unsafe { bindings::sum(cuda_column.device_ptr, size) };
        println!("value {:?}", value);
        println!("host {:?}", base_field_cuda_column_to_vec(&cuda_column));
    }
}
