const CUDA_LIB_DIR: &str = "/workspaces/cuda-rust-example/cuda";

fn main() {
    println!(
        "cargo:rerun-if-changed={}/src/batch_inverse.cu",
        CUDA_LIB_DIR
    );
    println!("cargo:rerun-if-changed={}/src/bit_reverse.cu", CUDA_LIB_DIR);
    println!("cargo:rerun-if-changed={}/src/fields.cu", CUDA_LIB_DIR);
    println!("cargo:rerun-if-changed={}/src/utils.cu", CUDA_LIB_DIR);

    println!(
        "cargo:rerun-if-changed={}/src/batch_inverse.cuh",
        CUDA_LIB_DIR
    );
    println!(
        "cargo:rerun-if-changed={}/src/bit_reverse.cuh",
        CUDA_LIB_DIR
    );
    println!("cargo:rerun-if-changed={}/src/fields.cuh", CUDA_LIB_DIR);
    println!("cargo:rerun-if-changed={}/src/utils.cuh", CUDA_LIB_DIR);

    println!("cargo:rustc-link-search={}", CUDA_LIB_DIR);

    let status = std::process::Command::new("nvcc")
        .args(&[
            "-arch=sm_50",
            "-Xcompiler",
            "-fPIC",
            "-shared",
            "-o",
            &format!("{}/libgpubackend.so", CUDA_LIB_DIR),
            &format!("{}/src/batch_inverse.cu", CUDA_LIB_DIR),
            &format!("{}/src/bit_reverse.cu", CUDA_LIB_DIR),
            &format!("{}/src/fields.cu", CUDA_LIB_DIR),
            &format!("{}/src/utils.cu", CUDA_LIB_DIR),
        ])
        .status()
        .expect("Failed to execute nvcc");
    if !status.success() {
        panic!("nvcc failed with status: {}", status);
    }
}
