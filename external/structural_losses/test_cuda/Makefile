nvcc=/usr/local/cuda/bin/nvcc
cudalib=/usr/local/cuda/lib64

nvcc_flags=--expt-relaxed-constexpr

# does not work to remove the "missing return statement at end of void function" warnings, but let's just keep it there
x_flags=-Xcompiler -Wno-return-type -Xcompiler -fPIC -O2

all: multiemd

multiemd: multiemd.cu.o multiemd.cpp sort_indices.c.o
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 --device-link -o multiemd_link.o multiemd.cu.o sort_indices.c.o -I . -DGOOGLE_CUDA=1 
	g++ -o multiemd -std=c++11 multiemd.cpp multiemd.cu.o sort_indices.c.o multiemd_link.o -I . -I /usr/local/cuda/include -lcudart -L $(cudalib) -O2 -D_GLIBCXX_USE_CXX11_ABI=0 

multiemd.cu.o: multiemd.cu 
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 --device-c -o multiemd.cu.o multiemd.cu -I . -DGOOGLE_CUDA=1 -x cu $(x_flags) $(nvcc_flags)

sort_indices.c.o: sort_indices.c 
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 --device-c -o sort_indices.c.o sort_indices.c -I . -DGOOGLE_CUDA=1 -x cu $(x_flags) $(nvcc_flags)

clean:
	rm -f *.o 
	rm -rf *.so

# Remove built-in rules (or Makefile will think it knows how to create .cu.o files from source files, here .cu).
.SUFFIXES:

.PHONY: all clean
