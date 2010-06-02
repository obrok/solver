all: main

main: main.obj solver.obj kernels.obj
	nvcc -o main -O2 main.obj solver.obj kernels.obj
	./main.exe

main.obj: main.cpp
	nvcc -c -O2 main.cpp
	
solver.obj: solver.cu solver.h
	nvcc -c -O2 solver.cu

kernels.obj: kernels.cu kernels.h
	nvcc -c -O2 kernels.cu
