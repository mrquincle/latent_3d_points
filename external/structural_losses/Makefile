# Configured for Google Colab

nvcc=/usr/local/cuda/bin/nvcc
cudalib=/usr/local/cuda/lib64

# !find / -name cuda_config.h only returns 3.6 lib
#
#
python=python3.6

tensorflow=/usr/local/lib/$(python)/dist-packages/tensorflow

# Todo: make above path to be found without hardcoding it, something like this:
# tensorflow = $($(python) -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

flags=
nvcc_flags=--expt-relaxed-constexpr

# does not work to remove the "missing return statement at end of void function" warnings, but let's just keep it there
x_flags=-Xcompiler -Wno-return-type

ld_flags=-L$(tensorflow) -ltensorflow_framework

all: tf_approxmatch_so.so tf_nndistance_so.so

tf_approxmatch_so.so: tf_approxmatch_g.cu.o tf_approxmatch.cpp
	g++ -std=c++11 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so -shared -fPIC -I $(tensorflow)/include -lcudart -L $(cudalib) $(ld_flags) -O2 -D_GLIBCXX_USE_CXX11_ABI=0 $(flags)

tf_approxmatch_g.cu.o: tf_approxmatch_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_approxmatch_g.cu.o tf_approxmatch_g.cu -I $(tensorflow)/include -DGOOGLE_CUDA=1 -x cu $(x_flags) -Xcompiler -fPIC -O2 $(nvcc_flags)

tf_nndistance_so.so: tf_nndistance_g.cu.o tf_nndistance.cpp
	g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I $(tensorflow)/include -lcudart -L $(cudalib) $(ld_flags) -O2 -D_GLIBCXX_USE_CXX11_ABI=0 $(flags)

tf_nndistance_g.cu.o: tf_nndistance_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I $(tensorflow)/include -DGOOGLE_CUDA=1 -x cu $(x_flags) -Xcompiler -fPIC -O2 $(nvcc_flags)


clean:
	rm -f tf_approxmatch_so.so
	rm -f tf_nndistance_so.so
	rm -f *.cu.o 

# Remove built-in rules (or Makefile will think it knows how to create .cu.o files from source files, here .cu).
.SUFFIXES:

.PHONY: all clean
