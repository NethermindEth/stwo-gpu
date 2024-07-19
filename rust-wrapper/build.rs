const CUDA_LIB_DIR: &str = "/workspaces/cuda-rust-example/libexample";

fn main() {
    println!("cargo:rerun-if-changed={}/src/example.cu", CUDA_LIB_DIR);

    let status = std::process::Command::new("nvcc")
        .args(&[
            "-arch=sm_50",
            "-Xcompiler",
            "-fPIC",
            "-shared",
            "-o",
            &format!("{}/libexample.so", CUDA_LIB_DIR),
            &format!("{}/src/example.cu", CUDA_LIB_DIR),
        ])
        .status()
        .expect("Failed to execute nvcc");
    if !status.success() {
        panic!("nvcc failed with status: {}", status);
    }

    println!("cargo:rustc-link-search={}", CUDA_LIB_DIR);
}

