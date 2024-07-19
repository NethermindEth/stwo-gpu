#[link(name = "gpubackend")]
extern "C" {
    pub fn generate_array(a: u32) -> *const u32;
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn sum(a: *const u32, size: u32) -> u32;
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn m31_device_to_host(device_ptr: *const u32, host_ptr: *const u32, size: u32) -> u32;
}
