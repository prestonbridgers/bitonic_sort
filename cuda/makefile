NVCC = /usr/bin/nvcc
CC = g++

DEBUGFLAGS=
#DEBUGFLAGS=-D DEBUG

#No optmization flags
#--compiler-options sends option to host compiler; -Wall is all warnings
#NVCCFLAGS = -c --compiler-options -Wall

#Optimization flags: -O2 gets sent to host compiler; -Xptxas -O2 is for
#optimizing PTX 
NVCCFLAGS = -c -O2 -Xptxas -O2 --compiler-options -Wall 

#Flags for debugging
#NVCCFLAGS = -c -G --compiler-options -Wall --compiler-options -g

OBJS = wrappers.o CudaBitonicSort.o
.SUFFIXES: .cu .o .h 
.cu.o:
	$(NVCC) $(DEBUGFLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS) $< -o $@

sort: $(OBJS)
	$(CC) $(OBJS) -L/usr/local/cuda/lib64 -lm -lcuda -lcudart -o $@

CudaBitonicSort.o: CudaBitonicSort.cu helpers.h wrappers.h

wrappers.o: wrappers.cu wrappers.h

clean:
	rm sort *.o
