use cmake;

fn main() {
    // Build cuda code
    println!("cargo:rerun-if-changed=../cuda/CMakeLists.txt");
    println!("cargo:rerun-if-changed=../cuda/src");
    println!("cargo:rerun-if-changed=../cuda/include");

    let destination = cmake::build("../cuda/");

    println!("cargo:rustc-link-search={}/lib/", destination.display());
}
