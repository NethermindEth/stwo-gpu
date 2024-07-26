const CUDA_LIB_DIR: &str = "/workspaces/stwo-gpu/cuda";

fn main() {
    // TODO: running CMake

    // Build cuda code
    println!("cargo:rustc-link-search={}", CUDA_LIB_DIR);
}
