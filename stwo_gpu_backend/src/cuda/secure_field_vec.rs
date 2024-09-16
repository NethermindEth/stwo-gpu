use std::{sync::Arc};

use stwo_prover::core::fields::qm31::SecureField;

use super::{bindings, cuda_memory::CudaMemory};

#[derive(Debug)]
pub struct SecureFieldVec {
    pub(crate) memory: Arc<CudaMemory>,
    pub(crate) offset: usize,
    pub(crate) size: usize,
}

impl SecureFieldVec {
    pub fn new_uninitialized(size: usize) -> Self {
        Self {
            memory: Arc::new(CudaMemory::new_uninitialized(size * 4)),
            offset: 0,
            size
        }
    }

    pub unsafe fn as_ptr(&self) -> *mut u32 {
        self.memory.as_ptr().offset((self.offset * 4) as isize)
    }

    pub fn split_at(self, offset: usize) -> (Self, Self) {
        let left_chunk = Self { memory: self.memory.clone(), offset: 0, size: offset};
        let right_chunk = Self { memory: self.memory, offset, size: self.size - offset};
        (left_chunk, right_chunk)
    }

    pub fn from_vec(host_array: Vec<SecureField>) -> Self {
        let secure_field_vec = Self::new_uninitialized(host_array.len());
        unsafe {
            bindings::copy_uint32_t_vec_from_host_to_device(
                host_array.as_ptr() as *const u32,
                secure_field_vec.as_ptr(),
                (host_array.len() * 4) as u32
            );
        }
        secure_field_vec
    }

    pub fn new_zeroes(size: usize) -> Self {
        Self {
            memory: Arc::new(CudaMemory::new_zeroes(size * 4)),
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
                (other.size * 4) as u32,
            );
        }
    }

    pub fn to_vec(&self) -> Vec<SecureField> {
        let mut host_data: Vec<SecureField> = Vec::with_capacity(self.size);
        unsafe {
            host_data.set_len(self.size);
            bindings::copy_uint32_t_vec_from_device_to_host(
                self.as_ptr(),
                host_data.as_mut_ptr() as *mut u32,
                (self.size * 4) as u32,
            );
        }
        host_data
    }
}

impl Clone for SecureFieldVec {
    fn clone(&self) -> Self {
        let mut cloned = Self::new_uninitialized(self.size);
        cloned.copy_from(self);
        cloned
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stwo_prover::core::fields::qm31::SecureField;

    #[test]
    fn test_constructor() {
        let size = 1 << 5;
        let from_raw = (1..(size + 1) as u32).collect::<Vec<u32>>();
        let host_data = from_raw
            .chunks(4)
            .map(|a| SecureField::from_u32_unchecked(a[0], a[1], a[2], a[3]))
            .collect::<Vec<_>>();
        let secure_field_vec = SecureFieldVec::from_vec(host_data.clone());

        assert_eq!(secure_field_vec.to_vec(), host_data);
        assert_eq!(secure_field_vec.size, host_data.len());
    }
}
