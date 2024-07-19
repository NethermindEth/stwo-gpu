#[link(name = "example")]
extern "C" {
    pub(crate) fn generate_array(a: i32) -> * const i32;
}

#[link(name = "example")]
extern "C" {
    pub(crate) fn sum(a: * const i32, size: i32) -> i32;
}

fn main() {
    let size = 1 << 11;
    let array = unsafe { generate_array(size) };
    let value = unsafe { sum(array, size) };
    println!("{:?}", value);
}

