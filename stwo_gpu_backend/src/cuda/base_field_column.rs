use stwo_prover::core::{backend::Column, fields::m31::BaseField};

use crate::cuda::bindings;

#[derive(Clone, Debug)]
pub struct BaseFieldVec {
    pub(crate) device_ptr: *const u32,
    pub(crate) size: usize,
}

impl BaseFieldVec {
    pub fn to_vec(&self) -> Vec<BaseField> {
        let mut host_data: Vec<BaseField> = Vec::with_capacity(self.size);
        unsafe {
            host_data.set_len(self.size.try_into().unwrap());
            bindings::m31_device_to_host(
                self.device_ptr,
                host_data.as_mut_ptr() as *const u32,
                self.size as u32,
            );
        }
        host_data
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1() {
        let size = 1 << 3;

        let cuda_column = BaseFieldVec {
            device_ptr: unsafe { bindings::generate_array(size) },
            size: size as usize,
        };

        let value = unsafe { bindings::sum(cuda_column.device_ptr, size) };
        println!("value {:?}", value);
        println!("host {:?}", cuda_column.to_cpu());
    }
}
