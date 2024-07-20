#[link(name = "gpubackend")]
extern "C" {
    pub fn copy_uint32_t_vec_from_device_to_host(
        device_ptr: *const u32,
        host_ptr: *const u32,
        size: u32,
    );
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn copy_uint32_t_vec_from_host_to_device(host_ptr: *const u32, size: u32) -> *const u32;
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn free_uint32_t_vec(device_ptr: *const u32);
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn bit_reverse_base_field(array: *const u32, size: usize, bits: usize);
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn bit_reverse_secure_field(array: *const u32, size: usize, bits: usize);
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn batch_inverse_base_field(
        from: *const u32,
        dst: *const u32,
        size: usize,
        log_size: usize,
    );
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn batch_inverse_secure_field(
        from: *const u32,
        dst: *const u32,
        size: usize,
        log_size: usize,
    );
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn sort_values(from: *const u32, size: usize) -> *const u32;
}
