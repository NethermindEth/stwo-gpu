use cmake;

fn main() {
    println!("cargo:rerun-if-changed=../cuda/CMakeLists.txt");
    println!("cargo:rerun-if-changed=../cuda/src");
    println!("cargo:rerun-if-changed=../cuda/include");

    let destination = cmake::build("../cuda/");

    // Build cuda code
    println!("cargo:rustc-link-search={}/lib/", destination.display());
}
