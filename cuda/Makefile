objects = build/batch_inverse.o build/bit_reverse.o build/circle.o build/fields.o build/fri.o build/utils.o
cuda = src/batch_inverse.cu src/bit_reverse.cu src/circle.cu src/fields.cu src/fri.cu src/utils.cu

all: $(objects)
	nvcc -Xcompiler -fPIC -shared -arch=sm_50 $(objects) -o libgpubackend.so

build/%.o: src/%.cu
	mkdir -p build/
	nvcc -Xcompiler -fPIC -x cu -arch=sm_50 -I. -dc $< -o $@

clean:
	rm -rf build/ libgpubackend.so