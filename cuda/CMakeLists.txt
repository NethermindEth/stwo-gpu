cmake_minimum_required(VERSION 3.22)
project(stwo-gpu DESCRIPTION "stwo gpu backend implementation")

set(CMAKE_CXX_STANDARD 14)

enable_language("CUDA")

include_directories(include)

add_library(gpubackend SHARED
        src/batch_inverse.cu
        src/bit_reverse.cu
        src/circle.cu
        src/utils.cu
        src/accumulate.cu
        src/fri.cu
        src/fields.cu
        src/blake2s.cu
        src/quotients.cu
        src/example.cu
        src/mle.cu
)
target_compile_options(gpubackend PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -arch=sm_50;
        -I.;
        -dc;
        >)
set_target_properties(gpubackend PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

install(TARGETS gpubackend
        LIBRARY DESTINATION lib/)
