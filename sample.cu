/********************************************************************
*  sample.cu
*  This is a example of the CUDA program.
*********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cutil_inline.h>

/************************************************************************/
/* Init CUDA                                                            */
/************************************************************************/
#if __DEVICE_EMULATION__

bool InitCUDA(void){return true;}

#else
bool InitCUDA(void)
{
	int count = 0;
	int i = 0;

	cudaGetDeviceCount(&count);
	if(count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}

	for(i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if(prop.major >= 1) {
				break;
			}
		}
	}
	if(i == count) {
		fprintf(stderr, "There is no device supporting CUDA.\n");
		return false;
	}
	cudaSetDevice(i);

	printf("CUDA initialized.\n");
	return true;
}

#endif
/************************************************************************/
/* Example                                                              */
/************************************************************************/
__global__ static void HelloCUDA(char* result, int num)
{
	int i = 0;
	char p_HelloCUDA[] = "Hello CUDA!";
	for(i = 0; i < num; i++) {
		result[i] = p_HelloCUDA[i];
	}
}

/************************************************************************/
/* HelloCUDA                                                            */
/************************************************************************/
int TestHelloCUDA(void)
{

	if(!InitCUDA()) {
		return 0;
	}

	char	*device_result	= 0;
	char	host_result[12]	={0};

	cutilSafeCall( cudaMalloc((void**) &device_result, sizeof(char) * 11));

	unsigned int timer = 0;
	cutilCheckError( cutCreateTimer( &timer));
	cutilCheckError( cutStartTimer( timer));

	HelloCUDA<<<1, 1, 0>>>(device_result, 11);
	cutilCheckMsg("Kernel execution failed\n");

	cudaThreadSynchronize();
	cutilCheckError( cutStopTimer( timer));
	printf("Processing time: %f (ms)\n", cutGetTimerValue( timer));
	cutilCheckError( cutDeleteTimer( timer));

	cutilSafeCall( cudaMemcpy(host_result, device_result, sizeof(char) * 11, cudaMemcpyDeviceToHost));
	printf("%s\n", host_result);

	cutilSafeCall( cudaFree(device_result));

	return 0;
}
