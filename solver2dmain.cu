#include <stdio.h>
#include <cuda.h>
#include <windows.h>

#include "solver.h"

void reduce(matrix* oldMatrices, int n, matrix* newMatrices);
void solve(matrix* matrices, int n);

int main(){
	matrix* matrices[LOGN+1];
	
	LARGE_INTEGER l;
	LARGE_INTEGER l2;

	
	cudaMalloc((void**)&(matrices[0]), sizeof(matrix)*N);
	zero<<<N*N, N>>>(matrices[0]);
	
	fillInside<<<2*N,(N-2)>>>(matrices[0]+1);
	fillLeft<<<2, N>>>(matrices[0]);
	fillRight<<<2, N>>>(matrices[0]+N-1);
	cudaThreadSynchronize();
	
	QueryPerformanceCounter(&l);
	
	int i,j;
	int n ;		
	for(i = 0,n=N; i < LOGN; i++,n/=2){
		cudaMalloc((void**)&matrices[i+1], sizeof(matrix)*n/2);
		zero<<<n*n/4, N>>>(matrices[i+1]);
		
		reduce(matrices[i], n, matrices[i+1]);
	}
	
	for(i = LOGN-1,n=2; i >= 0; i--,n*=2){
		solve(matrices[i], n);
		if(i > 0)
			for(j = 0; j < n; j+=2){
				copyBLower<<<1, N>>>(matrices[i]+j, matrices[i]+j+1, matrices[i-1]+j*2+1);
				copyBUpper<<<1, N>>>(matrices[i]+j, matrices[i]+j+1, matrices[i-1]+j*2+2);
			}
		cudaThreadSynchronize();
	}
	
	QueryPerformanceCounter(&l2);
	
	printf("%d %d\n", N, l2.u.LowPart-l.u.LowPart);
	
}

void reduce(matrix* oldMatrices, int n, matrix* newMatrices){
	int i;
	for(i = 0; i < n; i += 2){
		copyUpperLeft<<<N,N>>>(oldMatrices+i, newMatrices+i/2);
		copyLowerRight<<<N,N>>>(oldMatrices+i+1, newMatrices+i/2);
		copyBUpper<<<1,N>>>(oldMatrices+i, newMatrices+i/2);
		copyBLower<<<1,N>>>(oldMatrices+i+1, newMatrices+i/2);
	}
	
	float* coeffs;
		cudaMalloc((void**)&coeffs, sizeof(float)*3*N*n);
	zero<<<3*n, N>>>(coeffs);
	
	cudaThreadSynchronize();
		
	int row;
	for(row = 0; row < N; row++){
		float* dElement;
		cudaMalloc((void**)&dElement, sizeof(float));
		calculateElement<<<1,1>>>(dElement, row, oldMatrices+i, oldMatrices+i+1);
		cudaThreadSynchronize();
		
		//Calculate coefficients
		for(i = 0; i < n; i += 2){
			countCoeffsUpper<<<1, N>>>(dElement, oldMatrices+i, row, coeffs+(3*N)*i);
			countCoeffsCenter<<<1,N-row-1>>>(dElement, oldMatrices+i, oldMatrices+i+1, row, coeffs+(3*N)*i+N);
			countCoeffsLower<<<1, N>>>(dElement, oldMatrices+i+1, row, coeffs+(3*N)*i+2*N);
		}
		cudaThreadSynchronize();
		
		for(i = 0; i < n; i += 2){
			updateUpperLeft<<<N, N>>>(row, oldMatrices+i, newMatrices+i/2, coeffs+(3*N)*i);
			updateUpperCenter<<<N, N>>>(row, oldMatrices+i, oldMatrices+i+1, coeffs+(3*N)*i);
			updateUpperRight<<<N, N>>>(row, oldMatrices+i+1, newMatrices+i/2, coeffs+(3*N)*i);
			updateBUpper<<<1, N>>>(row, oldMatrices+i, oldMatrices+i+1, newMatrices+i/2, coeffs+(3*N)*i);
			
			updateLeftCenter<<<N-row-1, N>>>(row, oldMatrices+i, coeffs+(3*N)*i+N);
			updateCenter<<<N-row-1, N>>>(row, oldMatrices+i, oldMatrices+i+1, coeffs+(3*N)*i+N);
			updateRightCenter<<<N-row-1, N>>>(row, oldMatrices+i+1, coeffs+(3*N)*i+N);
			updateBCenter<<<1, N-row-1>>>(row, oldMatrices+i, oldMatrices+i+1, coeffs+(3*N)*i+N);
			
			updateLowerLeft<<<N, N>>>(row, oldMatrices+i, newMatrices+i/2, coeffs+(3*N)*i+2*N);
			updateLowerCenter<<<N, N>>>(row, oldMatrices+i, oldMatrices+i+1, coeffs+(3*N)*i+2*N);
			updateLowerRight<<<N, N>>>(row, oldMatrices+i+1, newMatrices+i/2, coeffs+(3*N)*i+2*N);
			updateBLower<<<1, N>>>(row, oldMatrices+i, oldMatrices+i+1, newMatrices+i/2, coeffs+(3*N)*i+2*N);
			
		}
		
		cudaThreadSynchronize();
	}
	
	cudaFree(coeffs);
}

void solve(matrix* matrices, int n){
	int i, j;
	
	for(i = 0; i < N; i++){
		for(j = 0; j < n; j+=2)
			backwardsSubstitutionRight<<<1, N>>>(i, matrices+j, matrices+j+1);
		cudaThreadSynchronize();
		for(j = 0; j < n; j+=2)
			backwardsSubstitutionLeft<<<1,N>>>(i, matrices+j, matrices+j+1);
		cudaThreadSynchronize();
	}
	
	for(i = 0; i < N; i++){
		for(j = 0; j < n; j+=2)
			backwardsSubstitutionCenter<<<1, N-i-1>>>(N-i-1, matrices+j, matrices+j+1);
		cudaThreadSynchronize();
	}
	
}