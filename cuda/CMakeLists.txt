cmake_minimum_required(VERSION 3.22)
project(stwo-gpu DESCRIPTION "stwo gpu backend implementation")

set(CMAKE_CXX_STANDARD 14)

enable_language("CUDA")

include_directories(include)

add_library(gpubackend SHARED
        src/batch_inverse.cu
        src/bit_reverse.cu
        src/poly/eval_at_point.cu
        src/poly/ifft.cu
        src/poly/rfft.cu
        src/poly/twiddles.cu
        src/poly/utils.cu
        src/fri/fold_line.cu
        src/fri/fold_circle_into_line.cu
        src/fri/utils.cu
        src/utils.cu
        src/accumulate.cu
        src/fields.cu
        src/blake2s.cu
        src/quotients.cu
        src/example.cu
)
target_compile_options(gpubackend PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -arch=sm_50;
        -I.;
        -dc;
        >)
set_target_properties(gpubackend PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

install(TARGETS gpubackend
        LIBRARY DESTINATION lib/)
