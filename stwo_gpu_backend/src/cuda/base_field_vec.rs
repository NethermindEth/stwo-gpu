use std::{ffi::c_void, sync::Arc};

use stwo_prover::core::fields::m31::BaseField;

use super::{bindings, cuda_memory::CudaMemory};


#[derive(Debug)]
pub struct BaseFieldVec {
    pub(crate) memory: Arc<CudaMemory>,
    pub(crate) offset: usize,
    pub(crate) size: usize,
}

impl BaseFieldVec {
    pub fn new_uninitialized(size: usize) -> Self {
        let memory = Arc::new(CudaMemory::new_uninitialized(size));
        Self { memory, offset: 0, size }
    }

    pub unsafe fn as_ptr(&self) -> *mut u32 {
        self.memory.as_ptr().offset(self.offset as isize)
    }

    pub fn split_at(&self, length: usize) -> (Self, Self) {
        let left_chunk = Self {
            memory: self.memory.clone(),
            offset: self.offset,
            size:   length
        };
        let right_chunk = Self {
            memory: self.memory.clone(),
            offset: self.offset + length,
            size:   self.size - length
        };
        (left_chunk, right_chunk)
    }

    pub fn from_vec(host_array: Vec<BaseField>) -> Self {
        let base_field_vec = Self::new_uninitialized(host_array.len());
        unsafe {
            bindings::copy_uint32_t_vec_from_host_to_device(
                host_array.as_ptr() as *const u32,
                base_field_vec.as_ptr(),
                host_array.len() as u32
            );
        }
        base_field_vec
    }

    pub fn copy_from_vec(&mut self, host_array: &[BaseField]) {
        unsafe {
            bindings::copy_uint32_t_vec_from_host_to_device(
                host_array.as_ptr() as *const u32,
                self.as_ptr(),
                host_array.len() as u32
            );
        }
    }

    pub fn new_zeroes(size: usize) -> Self {
        Self {
            memory: Arc::new(CudaMemory::new_zeroes(size)),
            offset: 0,
            size
        }
    }

    pub fn copy_from(&mut self, other: &Self) {
        assert!(self.size >= other.size);
        unsafe {
            bindings::copy_uint32_t_vec_from_device_to_device(
                other.as_ptr(),
                self.as_ptr(),
                other.size as u32,
            );
        }
    }

    pub fn to_vec(&self) -> Vec<BaseField> {
        let mut host_data: Vec<BaseField> = Vec::with_capacity(self.size);
        unsafe {
            host_data.set_len(self.size);
            bindings::copy_uint32_t_vec_from_device_to_host(
                self.as_ptr(),
                host_data.as_mut_ptr() as *mut u32,
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

impl Drop for CudaMemory {
    fn drop(&mut self) {
        unsafe { bindings::cuda_free_memory(self.address.as_ptr() as *const c_void) };
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
        BaseFieldVec::new_zeroes(size);
        BaseFieldVec::new_zeroes(size);
        BaseFieldVec::new_zeroes(size);
        BaseFieldVec::new_zeroes(size);
        BaseFieldVec::new_zeroes(size);
        BaseFieldVec::new_zeroes(size);
        BaseFieldVec::new_zeroes(size);
        let new_zeroes = BaseFieldVec::new_zeroes(size);
        for a in new_zeroes.to_vec().iter() {
            assert_eq!(a, &BaseField::from(0));
        }
    }
}
