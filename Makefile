CC=g++
NVCC=nvcc

CUDAPATH=/usr/local/cuda
BUILD_DIR=build

CFLAGS = -c -m64 -I$(CUDAPATH)/include
NVCCFLAGS = -c -I$(CUDAPATH)/include

LFLAGS = -m64 -L$(CUDAPATH)/lib -lcuda -lcudart -lm

all: build clean

build: build_dir gpu cpu
	$(NVCC) $(LFLAGS) -o $(BUILD_DIR)/mainfun *.o

build_dir:
	mkdir -p $(BUILD_DIR)

gpu:
	$(NVCC) $(NVCCFLAGS) *.cu

cpu:			
	$(CC) $(CFLAGS) *.cpp

clean:
	rm *.o



