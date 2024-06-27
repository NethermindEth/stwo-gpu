mod accumulation;
mod bit_reverse;
mod circle;
pub mod column;
pub mod error;
mod fri;
pub mod m31;
pub mod qm31;
mod quotients;

use std::ffi::c_void;
use std::fmt::Debug;
use std::sync::Arc;

use cudarc::driver::{CudaDevice, DeviceRepr, LaunchConfig, ValidAsZeroBits};
// use error::Error;
use once_cell::sync::Lazy;

use super::Backend;
use crate::core::fields::m31::M31;
use crate::core::fields::qm31::QM31;

static DEVICE: Lazy<Arc<CudaDevice>> = Lazy::new(|| CudaDevice::new(0).unwrap().load());

type Device = Arc<CudaDevice>;

trait Load {
    fn load(self) -> Self;
}

impl Load for Device {
    fn load(self) -> Self {
        bit_reverse::load_bit_reverse_ptx(&self);
        m31::load_base_field(&self);
        qm31::load_secure_field(&self);
        column::load_batch_inverse_ptx(&self);
        circle::load_circle(&self);
        fri::load_fri(&self);
        self
    }
}

unsafe impl DeviceRepr for M31 {
    fn as_kernel_param(&self) -> *mut c_void {
        self as *const Self as *mut c_void
    }
}

unsafe impl DeviceRepr for QM31 {
    fn as_kernel_param(&self) -> *mut c_void {
        self as *const Self as *mut c_void
    }
}

unsafe impl DeviceRepr for &mut QM31 {
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        self as *const Self as *mut _
    }
}

unsafe impl ValidAsZeroBits for M31 {}
unsafe impl ValidAsZeroBits for QM31 {}

#[derive(Copy, Clone, Debug)]
pub struct GpuBackend;

impl Backend for GpuBackend {}

impl GpuBackend {
    /// Creates a [LaunchConfig] with:
    /// - block_dim == `num_threads`
    /// - grid_dim == `(n + num_threads - 1) / num_threads`
    /// - shared_mem_bytes == `shared_mem_bytes`
    pub fn launch_config_for_num_elems(
        n: u32,
        num_threads: u32,
        shared_mem_bytes: u32,
    ) -> LaunchConfig {
        let num_blocks = (n + num_threads - 1) / num_threads;
        LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (num_threads, 1, 1),
            shared_mem_bytes,
        }
    }
}
