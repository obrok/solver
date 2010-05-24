#include <stdio.h>
#include <cuda.h>

#include "solver.h"

__device__ inline int idx(){
	return blockDim.x*blockIdx.x+threadIdx.x;
}

__global__ void zero(matrix* A){
	matrix* myMatrix = A+idx()/(N*N);
	int myIndex = idx()%(N*N);
	myMatrix->ll[myIndex] = 0;
	myMatrix->lr[myIndex] = 0;
	myMatrix->ur[myIndex] = 0;
	myMatrix->ul[myIndex] = 0;
	myMatrix->ub[myIndex/N] = 0;
	myMatrix->lb[myIndex/N] = 0;
}

__global__ void zero(float* vec){
	vec[idx()] = 0;
}

__global__ void fillInside(matrix* insideMatrices){
	matrix* myMatrix = insideMatrices+(idx()/(2*N));
	int myRow = idx()%(2*N);
	
	if (myRow == 0 || myRow == N-1){
		myMatrix->ul[myRow*N+myRow] = 0.5;
		myMatrix->ub[myRow] = 10.0/N*(idx()/(2*N)+1);
	}else if (myRow == N || myRow == 2*N-1){
		myRow -= N;
		myMatrix->lr[myRow*N+myRow] = 0.5;
		myMatrix->lb[myRow] = 10.0/N*(idx()/(2*N)+2);
	}else if(myRow < N){
		myMatrix->ul[myRow*N+myRow] = -2.0;
		myMatrix->ul[myRow*N+myRow-1] = (float)1/2;
		myMatrix->ul[myRow*N+myRow+1] = (float)1/2;
		myMatrix->ur[myRow*N+myRow] = 1;
	}else{
		myRow -= N;
		myMatrix->lr[myRow*N+myRow] = -2.0;
		myMatrix->lr[myRow*N+myRow-1] = (float)1/2;
		myMatrix->lr[myRow*N+myRow+1] = (float)1/2;
		myMatrix->ll[myRow*N+myRow] = 1;
	}
}

__global__ void fillLeft(matrix* leftMatrix){
	int myRow = idx()%(2*N);
	
	if(myRow < N){
		leftMatrix->ul[myRow*N+myRow] = 1;
		leftMatrix->ub[myRow] = 0;
	}
	else if(myRow == N || myRow == 2*N-1){
		myRow -= N;
		leftMatrix->lr[myRow*N+myRow] = 0.5;
		leftMatrix->lb[myRow] = 10.0/N;
	}else{
		myRow -= N;
		leftMatrix->lr[myRow*N+myRow] = -2.0;
		leftMatrix->lr[myRow*N+myRow-1] = (float)1/2;
		leftMatrix->lr[myRow*N+myRow+1] = (float)1/2;
		leftMatrix->ll[myRow*N+myRow] = 1;
	}
}

__global__ void fillRight(matrix* rightMatrix){
	int myRow = idx()%(2*N);
	
	if(myRow >= N){
		myRow -= N;
		rightMatrix->lr[myRow*N+myRow] = 1;
		rightMatrix->lb[myRow] = 20;
	}
	else if(myRow == 0 || myRow == N-1){
		rightMatrix->ul[myRow*N+myRow] = 0.5;
		rightMatrix->ub[myRow] = 10.0/N*(N-1);
	}else{
		rightMatrix->ul[myRow*N+myRow] = -2.0;
		rightMatrix->ul[myRow*N+myRow-1] = (float)1/2;
		rightMatrix->ul[myRow*N+myRow+1] = (float)1/2;
		rightMatrix->ur[myRow*N+myRow] = 1;
	}
}

__global__ void copyUpperLeft(matrix* A, matrix* C){
	C->ul[idx()] = A->ul[idx()];
}

__global__ void copyLowerRight(matrix* B, matrix* C){
	C->lr[idx()] = B->lr[idx()];
}

__global__ void copyBUpper(matrix* A, matrix* C){
	C->ub[idx()] = A->ub[idx()];
}

__global__ void copyBLower(matrix* B, matrix* C){
	C->lb[idx()] = B->lb[idx()];
}

__global__ void calculateElement(float* dElement, int row, matrix* A, matrix* B){
	*dElement = A->lr[row*(N+1)] + B->ul[row*(N+1)];
}

__global__ void countCoeffsUpper(float* dElement, matrix* A, int col, float* coeffs) {
	coeffs[idx()] = A->ur[idx()*N + col] / (*dElement);
}

__global__ void countCoeffsLower(float* dElement, matrix* B, int col, float* coeffs) {
	coeffs[idx()] = B->ll[idx()*N + col] / (*dElement);
}

__global__ void countCoeffsCenter(float* dElement, matrix* A, matrix* B, int col, float* coeffs) {
	int newIdx = idx() + col + 1;
	int index = newIdx*N + col;

	coeffs[newIdx] = (A->lr[index] + B->ul[index]) / (*dElement);
}

__global__ void updateUpperLeft(int row, matrix* A, matrix* C, float* coeffs) {
	int x = idx() % N;
	int y = idx() / N;
	
	C->ul[idx()] = C->ul[idx()] - A->ll[row*N + x]*coeffs[y];
}

__global__ void updateLowerRight(int row, matrix* B, matrix* C, float* coeffs) {
	int x = idx() % N;
	int y = idx() / N;
	
	C->lr[idx()] = C->lr[idx()] - B->ur[row*N + x]*coeffs[y];
}

__global__ void updateLowerLeft(int row, matrix* A, matrix* C, float* coeffs) {
	int x = idx() % N;
	int y = idx() / N;
	
	C->ll[idx()] = C->ll[idx()] - A->ll[row*N + x] * coeffs[y];
}

__global__ void updateUpperRight(int row, matrix* B, matrix* C, float* coeffs) {
	int x = idx() % N;
	int y = idx() / N;
	
	C->ur[idx()] = C->ur[idx()] - B->ur[row*N + x] * coeffs[y];
}

__global__ void updateUpperCenter(int row, matrix* A, matrix* B, float* coeffs) {
	int x = idx() % N;
	int y = idx() / N;

	int index = (row * N) + x;
	A->ur[idx()] = A->ur[idx()] - (A->lr[index] + B->ul[index]) * coeffs[y];
}

__global__ void updateLowerCenter(int row, matrix* A, matrix* B, float* coeffs) {
	int x = idx() % N;
	int y = idx() / N;

	int index = (row * N) + x;
	B->ll[idx()] = B->ll[idx()] - (A->lr[index] + B->ul[index]) * coeffs[y];
}

__global__ void updateCenter(int row, matrix* A, matrix* B, float* coeffs) {
	int newIdx = idx() + (row + 1) * N;
	int x = newIdx % N;
	int y = newIdx / N;
	
	int index = (row * N) + x;
	float element = A->lr[newIdx] + B->ul[newIdx] - (A->lr[index] + B->ul[index]) * coeffs[y];
	
	A->lr[newIdx] = element / 2;
	B->ul[newIdx] = element / 2;
}

__global__ void updateLeftCenter(int row, matrix* A, float* coeffs) {
	int newIdx = idx() + (row + 1) * N;
	int x = newIdx % N;
	int y = newIdx / N;
	
	A->ll[newIdx] = A->ll[newIdx] - A->ll[(row * N) + x] * coeffs[y];
}

__global__ void updateRightCenter(int row, matrix* B, float* coeffs) {
	int newIdx = idx() + (row + 1) * N;
	int x = newIdx % N;
	int y = newIdx / N;

	B->ur[newIdx] = B->ur[newIdx] - B->ur[(row * N) + x] * coeffs[y];
}

__global__ void updateBCenter(int row, matrix* A, matrix* B, float* coeffs) {
	int newIdx = row + 1 + idx();
	float element = A->lb[newIdx] + B->ub[newIdx] - (A->lb[row] + B->ub[row])*coeffs[newIdx];
	A->lb[newIdx] = element / 2;
	B->ub[newIdx] = element / 2;
}

__global__ void updateBLower(int row, matrix* A, matrix* B, matrix* C, float* coeffs) {
	int newIdx = idx();
	C->lb[newIdx] = C->lb[newIdx] - (B->ub[row] + A->lb[row]) * coeffs[newIdx];
}

__global__ void updateBUpper(int row, matrix* A, matrix* B, matrix* C, float* coeffs) {
	int newIdx = idx();
	
	C->ub[newIdx] = C->ub[newIdx] - (B->ub[row] + A->lb[row]) * coeffs[newIdx];
}

__global__ void backwardsSubstitutionRight(int col, matrix* A, matrix* B) {
	int index = idx();
	float element = A->lb[index] + B->ub[index] - B->ur[index*N + col] * B->lb[col] ;// B->lr[col*N+col];
	A->lb[index] = element/2;
	B->ub[index] = element/2;
}

__global__ void backwardsSubstitutionLeft(int col, matrix* A, matrix* B) {
	int index = idx();
	float element = A->lb[index] + B->ub[index] - B->ll[index*N + col] * A->ub[col] ;// A->ul[col*N+col];
	A->lb[index] = element/2;
	B->ub[index] = element/2;
}

__global__ void backwardsSubstitutionCenter(int col, matrix* A, matrix* B) {
	int index = idx();
	int index2d = index*N + col;
	int indexCoeff = col*N + col;
	float element = A->lb[index]+B->ub[index] - (A->lb[col]+B->ub[col])*(A->lr[index2d]+B->ul[index2d])/(A->lr[indexCoeff]+B->ul[indexCoeff]);
	A->lb[index] = element/2;
	B->ub[index] = element/2;
}

__global__ void copyBLower(matrix* A, matrix* B, matrix* C) {
	int index = idx();
	C->lb[index] = (A->lb[index] + B->ub[index])/(A->lr[index*(N+1)] + B->ul[index*(N+1)]);
}

__global__ void copyBUpper(matrix* A, matrix* B, matrix* C) {
	int index = idx();
	C->ub[index] = (A->lb[index] + B->ub[index])/(A->lr[index*(N+1)] + B->ul[index*(N+1)]);
}

void printDeviceMatrix(matrix* deviceMatrix){
	matrix temp;
	cudaMemcpy(&temp, deviceMatrix, sizeof(matrix), cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
	
	printHostMatrix(&temp);
}

void printHostMatrix(matrix* temp) {
	printf("ul\n");
	printFloatArray(temp->ul, N);
	printf("ur\n");
	printFloatArray(temp->ur, N);
	printf("ll\n");
	printFloatArray(temp->ll, N);
	printf("lr\n");
	printFloatArray(temp->lr, N);
	printf("ub\n");
	printHostVector(temp->ub, N);
	printf("lb\n");
	printHostVector(temp->lb, N);
}

void printFloatArray(float *M, int size) {
	int i,j;
	for(i = 0; i < size; i++){
		for(j = 0; j < size; j++) {
			printf("%f ", M[i*size + j]);
		}
		printf("\n");
	}
}

void printDeviceVector(float* dVec, int len){
	float* temp = (float*)malloc(sizeof(float)*len);
	cudaMemcpy(temp, dVec, sizeof(float)*len, cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
	
	printHostVector(temp, len);
	
	free(temp);
}

void printHostVector(float* V, int len) {
	int i;
	for(i = 0; i < len; i++)
		printf("%f ", V[i]);
	printf("\n");
}