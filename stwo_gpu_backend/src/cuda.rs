use crate::{backend::CudaBackend, column::BaseFieldCudaColumn};
use stwo_prover::core::{
    backend::{Column, ColumnOps},
    fields::{m31::BaseField, qm31::SecureField},
};

#[link(name = "gpubackend")]
extern "C" {
    pub fn generate_array(a: u32) -> *const u32;
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn sum(a: *const u32, size: u32) -> u32;
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn m31_device_to_host(device_ptr: *const u32, host_ptr: *const u32, size: u32) -> u32;
}

pub fn base_field_cuda_column_to_vec(column: &BaseFieldCudaColumn) -> Vec<BaseField> {
    let mut host_data: Vec<BaseField> = Vec::with_capacity(column.size);
    unsafe {
        host_data.set_len(column.size.try_into().unwrap());
        m31_device_to_host(
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
    use crate::{backend::CudaBackend, column::BaseFieldCudaColumn};
    use stwo_prover::core::{
        backend::{Column, ColumnOps},
        fields::{m31::BaseField, qm31::SecureField},
    };

    #[test]
    fn test_1() {
        let size = 1 << 3;

        let cuda_column = BaseFieldCudaColumn {
            device_ptr: unsafe { generate_array(size) },
            size: size as usize,
        };

        let value = unsafe { sum(cuda_column.device_ptr, size) };
        println!("value {:?}", value);
        println!("host {:?}", base_field_cuda_column_to_vec(&cuda_column));
    }
}
