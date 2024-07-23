use stwo_prover::core::fields::m31::BaseField;

use super::bindings;

#[derive(Clone, Debug)]
pub struct BaseFieldVec {
    pub(crate) device_ptr: *const u32,
    pub(crate) size: usize,
}

impl BaseFieldVec {
    pub fn new(device_ptr: *const u32, size: usize) -> Self {
        Self { device_ptr, size }
    }

    pub fn from_vec(host_array: Vec<BaseField>) -> Self {
        let device_ptr = unsafe {
            bindings::copy_uint32_t_vec_from_host_to_device(
                host_array.as_ptr() as *const u32,
                host_array.len() as u32,
            )
        };
        let size = host_array.len();
        Self::new(device_ptr, size)
    }

    pub fn new_uninitialized(size: usize) -> Self {
        Self::new(unsafe { bindings::cuda_malloc_uint32_t(size as u32) }, size)
    }

    pub fn new_zeroes(size: usize) -> Self {
        Self::new(
            unsafe { bindings::cuda_alloc_zeroes_uint32_t(size as u32) },
            size,
        )
    }

    pub fn copy_from(&mut self, other: &Self) {
        assert!(self.size >= other.size);
        unsafe {
            bindings::copy_uint32_t_vec_from_device_to_device(
                other.device_ptr,
                self.device_ptr,
                other.size as u32,
            );
        }
    }

    pub fn to_vec(&self) -> Vec<BaseField> {
        let mut host_data: Vec<BaseField> = Vec::with_capacity(self.size);
        unsafe {
            host_data.set_len(self.size.try_into().unwrap());
            bindings::copy_uint32_t_vec_from_device_to_host(
                self.device_ptr,
                host_data.as_mut_ptr() as *const u32,
                self.size as u32,
            );
        }
        host_data
    }
}

impl Drop for BaseFieldVec {
    fn drop(&mut self) {
        unsafe { bindings::free_uint32_t_vec(self.device_ptr) };
    }
}

#[cfg(test)]
mod tests {
    use stwo_prover::core::fields::m31::BaseField;

    use super::*;

    #[test]
    fn test_constructor() {
        let size = 1 << 25;
        let host_data = (0..size).map(BaseField::from).collect::<Vec<_>>();
        let base_field_vec = BaseFieldVec::from_vec(host_data.clone());
        assert_eq!(base_field_vec.to_vec(), host_data);
        assert_eq!(base_field_vec.size, host_data.len());
    }
}
