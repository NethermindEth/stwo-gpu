use crate::cuda::bindings;
use std::{ffi::c_void, fmt::Debug};
use stwo_prover::core::vcs::blake2_hash::Blake2sHash;

#[derive(Debug)]
pub struct Blake2sHashVec {
    pub(crate) device_ptr: *const Blake2sHash,
    pub(crate) size: usize,
}

impl Blake2sHashVec {
    pub fn new(device_ptr: *const Blake2sHash, size: usize) -> Self {
        Self { device_ptr, size }
    }

    pub fn from_vec(host_array: Vec<Blake2sHash>) -> Self {
        let size = host_array.len();
        let device_ptr = unsafe {
            bindings::copy_blake_2s_hash_vec_from_host_to_device(host_array.as_ptr(), size)
        };
        Self::new(device_ptr, size)
    }

    pub fn new_uninitialized(size: usize) -> Self {
        Self::new(unsafe { bindings::cuda_malloc_blake_2s_hash(size) }, size)
    }

    pub fn new_zeroes(size: usize) -> Self {
        Self::new(
            unsafe { bindings::cuda_alloc_zeroes_blake_2s_hash(size) },
            size,
        )
    }

    pub fn copy_from(&mut self, other: &Self) {
        assert!(self.size >= other.size);
        unsafe {
            bindings::copy_blake_2s_hash_vec_from_device_to_device(
                other.device_ptr,
                self.device_ptr,
                other.size,
            );
        }
    }

    pub fn to_vec(&self) -> Vec<Blake2sHash> {
        let mut host_data: Vec<Blake2sHash> = Vec::with_capacity(self.size);
        unsafe {
            host_data.set_len(self.size);
            bindings::copy_blake_2s_hash_vec_from_device_to_host(
                self.device_ptr,
                host_data.as_mut_ptr(),
                self.size,
            );
        }
        host_data
    }
}

impl Clone for Blake2sHashVec {
    fn clone(&self) -> Self {
        let mut cloned = Self::new_uninitialized(self.size);
        cloned.copy_from(self);
        cloned
    }
}

impl Drop for Blake2sHashVec {
    fn drop(&mut self) {
        unsafe { bindings::cuda_free_memory(self.device_ptr as *const c_void) };
    }
}
