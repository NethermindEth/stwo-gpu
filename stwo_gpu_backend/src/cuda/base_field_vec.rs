use stwo_prover::core::fields::m31::BaseField;

use super::bindings;

#[derive(Debug)]
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

impl Clone for BaseFieldVec {
    fn clone(&self) -> Self {
        let mut cloned = Self::new_uninitialized(self.size);
        cloned.copy_from(self);
        cloned
    }
}

impl Drop for BaseFieldVec {
    fn drop(&mut self) {
        unsafe { bindings::free_uint32_t_vec(self.device_ptr) };
    }
}

// TODO (Daniel):: Impl Free
#[derive(Debug)]
pub struct VecBaseFieldVec {
    pub(crate) device_ptr: *const *const u32,
    pub(crate) col_size: usize,
    pub(crate) row_size: usize,
}

impl VecBaseFieldVec {
    pub fn from_vec(host_array: Vec<&BaseFieldVec>) -> Self {
        // Initialize host array
        let host_ptr = unsafe {
            bindings::unified_malloc_dbl_ptr_uint32_t(host_array.len())
        };

        // Set device pointers to host array
        host_array.iter().enumerate().for_each(|(i, bf)| unsafe {
            bindings::unified_set_dbl_ptr_uint32_t(host_ptr, bf.device_ptr, i)
        });

        // // Copy host array to device
        // let device_ptr = unsafe {
        //     bindings::cuda_set_dbl_ptr_uint32_t(host_ptr, host_array.len())
        // };

        Self {
            device_ptr: host_ptr,     
            col_size: host_array.len(),
            row_size: host_array[0].size, 
        }
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

    #[test]
    fn test_zeroes() {
        let size = 64;
        let new_zeroes = BaseFieldVec::new_zeroes(size);
        let new_zeroes = BaseFieldVec::new_zeroes(size);
        let new_zeroes = BaseFieldVec::new_zeroes(size);
        let new_zeroes = BaseFieldVec::new_zeroes(size);
        let new_zeroes = BaseFieldVec::new_zeroes(size);
        let new_zeroes = BaseFieldVec::new_zeroes(size);
        let new_zeroes = BaseFieldVec::new_zeroes(size);
        let new_zeroes = BaseFieldVec::new_zeroes(size);
        for a in new_zeroes.to_vec().iter() {
            assert_eq!(a, &BaseField::from(0));
        }
    }
}
