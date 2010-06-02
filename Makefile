all: japi emu

japi: main.obj solver.obj
	nvcc -o japi -O2 main.obj solver.obj
	./japi.exe
	
emu: emumain.obj emusolver.obj
	nvcc -deviceemu  -D_DEBUG -DWIN32 -D_CONSOLE -D_MBCS -Xcompiler /EHsc,/W3,/nologo,/Wp64,/Od,/Zi,/MTd -o emu emumain.obj emusolver.obj
	./emu.exe
	
main.obj: main.cu solver.h
	nvcc -c -O2 main.cu
emumain.obj: main.cu solver.h
	nvcc -deviceemu  -D_DEBUG -DWIN32 -D_CONSOLE -D_MBCS -Xcompiler /EHsc,/W3,/nologo,/Wp64,/Od,/Zi,/MTd -c main.cu -o emumain.obj

solver.obj: solver.cu solver.h
	nvcc -c -O2 solver.cu
emusolver.obj: solver.cu solver.h
	nvcc -deviceemu  -D_DEBUG -DWIN32 -D_CONSOLE -D_MBCS -Xcompiler /EHsc,/W3,/nologo,/Wp64,/Od,/Zi,/MTd -c solver.cu -o emusolver.obj	
