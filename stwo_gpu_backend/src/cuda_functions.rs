#[link(name = "gpubackend")]
extern "C" {
    pub fn generate_array(a: i32) -> *const i32;
}

#[link(name = "gpubackend")]
extern "C" {
    pub fn sum(a: *const i32, size: i32) -> i32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1() {
        let size = 1 << 11;
        let array = unsafe { generate_array(size) };
        let value = unsafe { sum(array, size) };
        println!("{:?}", value);
    }
}
