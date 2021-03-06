# Configured for Google Colab

nvcc=/usr/local/cuda/bin/nvcc
cudalib=/usr/local/cuda/lib64

# !find / -name cuda_config.h only returns 3.6 lib
#
#
python=python3.6

# Tensorflow 1.14.0
#tensorflow=/usr/local/lib/$(python)/dist-packages/tensorflow

# Tensorflow 1.15.0
tensorflow=/usr/local/lib/$(python)/dist-packages/tensorflow_core

# Todo: make above path to be found without hardcoding it, something like this:
# tensorflow = $($(python) -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

flags=
nvcc_flags=--expt-relaxed-constexpr

# does not work to remove the "missing return statement at end of void function" warnings, but let's just keep it there
x_flags=-Xcompiler -Wno-return-type -Xcompiler -fPIC -O2

# Check compiler/linker options for shared libraries if things can't be found with something like:
#   import tensorflow as tf
#   from __future__ import print_function
#   print(tf.sysconfig.get_compile_flags())
#   print(tf.sysconfig.get_link_flags())

ld_flags=-L$(tensorflow) -l:libtensorflow_framework.so.1

all: tf_multiemd_so.so tf_approxmatch_so.so tf_nndistance_so.so

tf_multiemd_so.so: sort_indices.c.o tf_multiemd_g.cu.o tf_multiemd.cpp 
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 --device-link -o tf_multiemd_link.o sort_indices.c.o tf_multiemd_g.cu.o -I . -I $(tensorflow)/include -DGOOGLE_CUDA=1 $(x_flags) $(nvcc_flags)
	g++ -std=c++11 tf_multiemd.cpp tf_multiemd_g.cu.o sort_indices.c.o tf_multiemd_link.o -o tf_multiemd_so.so -shared -fPIC -I . -I $(tensorflow)/include -lcudart -L $(cudalib) $(ld_flags) -O2 -D_GLIBCXX_USE_CXX11_ABI=0 $(flags)

tf_multiemd_g.cu.o: tf_multiemd_g.cu 
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 --device-c -o tf_multiemd_g.cu.o tf_multiemd_g.cu -I . -I $(tensorflow)/include -DGOOGLE_CUDA=1 -x cu $(x_flags) $(nvcc_flags)

sort_indices.c.o: sort_indices.c 
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 --device-c -o sort_indices.c.o sort_indices.c -I . -I $(tensorflow)/include -DGOOGLE_CUDA=1 -x cu $(x_flags) $(nvcc_flags)

tf_approxmatch_so.so: tf_approxmatch_g.cu.o tf_approxmatch.cpp
	g++ -std=c++11 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so -shared -fPIC -I $(tensorflow)/include -lcudart -L $(cudalib) $(ld_flags) -O2 -D_GLIBCXX_USE_CXX11_ABI=0 $(flags)

tf_approxmatch_g.cu.o: tf_approxmatch_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_approxmatch_g.cu.o tf_approxmatch_g.cu -I $(tensorflow)/include -DGOOGLE_CUDA=1 -x cu $(x_flags) $(nvcc_flags)

tf_nndistance_so.so: tf_nndistance_g.cu.o tf_nndistance.cpp
	g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I $(tensorflow)/include -lcudart -L $(cudalib) $(ld_flags) -O2 -D_GLIBCXX_USE_CXX11_ABI=0 $(flags)

tf_nndistance_g.cu.o: tf_nndistance_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I $(tensorflow)/include -DGOOGLE_CUDA=1 -x cu $(x_flags) $(nvcc_flags)


clean:
	rm -f *.so
	rm -f *.o 

# Remove built-in rules (or Makefile will think it knows how to create .cu.o files from source files, here .cu).
.SUFFIXES:

.PHONY: all clean
