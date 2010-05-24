all: japi rudy
japi: main.obj solver.obj
	nvcc -o japi -O2 main.obj solver.obj
	./japi.exe
	
japemu: emumain.obj emusolver.obj
	nvcc -deviceemu -o japi emumain.obj emusolver.obj
	japi > result.txt
	
rudy: emuAllTests.obj emuCuTest.obj emuSolverTest.obj emusolver.obj
	nvcc -deviceemu -o rudy emuAllTests.obj emuCuTest.obj emuSolverTest.obj emusolver.obj
	rudy

main.obj: main.cu solver.h
	nvcc -c -O2 main.cu
emumain.obj: main.cu solver.h
	nvcc -deviceemu -c main.cu -o emumain.obj

solver.obj: solver.cu solver.h
	nvcc -c -O2 solver.cu
emusolver.obj: solver.cu solver.h
	nvcc -deviceemu -c solver.cu -o emusolver.obj
	
SolverTest.obj:  SolverTest.cu solver.h
	nvcc -c SolverTest.cu
emuSolverTest.obj:  SolverTest.cu solver.h
	nvcc -deviceemu -c SolverTest.cu -o emuSolverTest.obj
	
CuTest.obj: CuTest.cu solver.h
	nvcc -c CuTest.cu
emuCuTest.obj: CuTest.cu solver.h
	nvcc -deviceemu -c CuTest.cu -o emuCuTest.obj
	
AllTests.obj: AllTests.cu solver.h
	nvcc -c AllTests.cu
emuAllTests.obj: AllTests.cu solver.h
	nvcc -deviceemu -c AllTests.cu -o emuAllTests.obj
