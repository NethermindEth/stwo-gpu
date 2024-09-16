use std::ptr::NonNull;

use super::bindings;


#[derive(Debug)]
pub(crate) struct CudaMemory {
    pub(crate) address: NonNull<u32>
}

impl CudaMemory {
    pub(crate) fn new_uninitialized(size: usize) -> Self {
        let device_ptr = unsafe {
            bindings::cuda_malloc_uint32_t(size as u32)
        };
        Self {
            address: NonNull::new(device_ptr).expect("Error initializing memory")
        }
    }

    pub(crate) fn new_zeroes(size: usize) -> Self {
        let device_ptr = unsafe {
                bindings::cuda_alloc_zeroes_uint32_t(size as u32)
        };
        Self {
            address: NonNull::new(device_ptr).expect("Error initializing memory")
        }
    }

    pub(crate) fn as_ptr(&self) -> *mut u32 {
        self.address.as_ptr()
    }

}