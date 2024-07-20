#[link(name = "gpubackend")]
extern "C" {
    pub fn generate_array(a: u32) -> *const u32;
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn last(a: *const u32, size: u32) -> u32;
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn copy_m31_vec_from_device_to_host(
        device_ptr: *const u32,
        host_ptr: *const u32,
        size: u32,
    );
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn copy_m31_vec_from_host_to_device(host_ptr: *const u32, size: u32) -> *const u32;
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
