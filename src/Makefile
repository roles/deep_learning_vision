CC=g++
NVCC=nvcc
INC_PATH=-I/usr/include/python2.7/ \
		 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ \
		 -I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy/
CPPFLAGS :=-g -O0 -fno-inline -std=c++0x -w -fPIC
NVCCFLAGS :=-G -g -w -Xcompiler -fPIC -arch=sm_20
CPPFLAGS += $(INC_PATH)
NVCCFLAGS += $(INC_PATH) 
LDFLAGS=-fPIC -shared -L/usr/local/cuda/lib -L/usr/local/cuda/lib64
OBJS=pyNVCRBM.o crbm.o crbm_kernel.o matrix.o nvmatrix.o

nvcrbm.so : $(OBJS)
	$(CC) $(LDFLAGS) $^ -lcudart -o $@

%.o: %.cpp
	$(CC) -c $(CPPFLAGS) $< -o $@

%.o: %.cu
	$(NVCC) -c $(NVCCFLAGS) $< -o $@

install64:
	cp nvcrbm.so /usr/lib/python2.7/dist-packages/nvcrbm.so

install:
	cp nvcrbm.so /usr/local/lib/python2.7/dist-packages/nvcrbm.so

clean:
	rm -rf *.so *.o
