use super::bindings;

#[derive(Clone, Debug)]
pub struct SecureFieldVec {
    pub(crate) device_ptr: *const u32,
    pub(crate) size: usize,
}

impl Drop for SecureFieldVec {
    fn drop(&mut self) {
        unsafe { bindings::free_uint32_t_vec(self.device_ptr) };
    }
}
