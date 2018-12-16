# Configured for Google Colab

nvcc = /usr/local/cuda/bin/nvcc
cudalib =  /usr/local/cuda/lib64
#tensorflow = /usr/local/lib/python3.6/dist-packages/tensorflow/include
tensorflow = /usr/local/lib/python2.7/dist-packages/tensorflow/include

flags=
nvcc_flags=--expt-relaxed-constexpr

# does not work, but whatever
x_flags=-Xcompiler -Wno-return-type

tf_lib=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

ld_flags=-L$(tf_lib) -ltensorflow_framework

#all: tf_approxmatch_so.so tf_approxmatch_g.cu.o tf_nndistance_so.so tf_nndistance_g.cu.o
all: tf_approxmatch_so.so tf_nndistance_so.so

tf_approxmatch_so.so: tf_approxmatch_g.cu.o tf_approxmatch.cpp
	g++ -std=c++11 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so -shared -fPIC -I $(tensorflow) -lcudart -L $(cudalib) $(ld_flags) -O2 -D_GLIBCXX_USE_CXX11_ABI=0 $(flags)

tf_approxmatch_g.cu.o: tf_approxmatch_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_approxmatch_g.cu.o tf_approxmatch_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu $(x_flags) -Xcompiler -fPIC -O2 $(nvcc_flags)

tf_nndistance_so.so: tf_nndistance_g.cu.o tf_nndistance.cpp
	g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I $(tensorflow) -lcudart -L $(cudalib) $(ld_flags) -O2 -D_GLIBCXX_USE_CXX11_ABI=0 $(flags)

tf_nndistance_g.cu.o: tf_nndistance_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu $(x_flags) -Xcompiler -fPIC -O2 $(nvcc_flags)


clean:
	rm tf_approxmatch_so.so
	rm tf_nndistance_so.so
	rm  *.cu.o 
