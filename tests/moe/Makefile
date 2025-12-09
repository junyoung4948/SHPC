CFLAGS=-std=c++14 -O3 -Wall -march=native -mavx2 -mfma -mno-avx512f -fopenmp -I/usr/local/cuda/include
CUDA_CFLAGS:=$(foreach option, $(CPPFLAGS),-Xcompiler=$(option) -arch=sm_75)

LDFLAGS=-pthread -L/usr/local/cuda/lib64
LDLIBS=-lstdc++ -lcudart -lm

CXX=g++
CUX=/usr/local/cuda/bin/nvcc

TARGET=main
OBJECTS=main.o util.o moe.o

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS) $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CFLAGS) -c -o $@ $^

%.o: %.cu
	$(CUX) $(CUDA_CFLAGS) -c -o $@ $^

clean:
	rm -rf $(TARGET) $(OBJECTS)
