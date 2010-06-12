all: japi 
japi: main.obj solver.obj
	nvcc -o japi -O2 main.obj solver.obj
	./japi.exe

gauss: gauss.obj solver.obj
	nvcc -o gauss gauss.obj solver.obj

main.obj: main.cu solver.h
	nvcc -c -O2 main.cu

solver.obj: solver.cu solver.h
	nvcc -c -O2 solver.cu

gauss.obj: gauss.cu
	nvcc -c -O2 gauss.cu
