#include <stdio.h>
#include <cuda.h>
#include "solver.h"
__device__ inline int idx(){
	return blockDim.x*blockIdx.x+threadIdx.x;
}

__host__ __device__ int matrix_size(int size) {
	return 4*size*size + 2*size;
}

__global__ void init_matrices(matrix* A, float* data, int size){
	A = A + idx();
	data = data + idx()*matrix_size(size);
	
	A->ur = data;
	A->ul = A->ur + size * size;
	A->lr = A->ul + size * size;
	A->ll = A->lr + size * size;
	A->ub = A->ll + size * size;
	A->lb = A->ub + size;
	
	for(int i = 0; i < size; i++) {
		for(int j = 0; j < size; j++) {
			A->ll[i*size + j] = 0;
			A->lr[i*size + j] = 0;
			A->ur[i*size + j] = 0;
			A->ul[i*size + j] = 0;
		}
		A->ub[i] = 0;
		A->lb[i] = 0;
	}
}
__global__ void init_vector(float* vec, int size){
	vec = vec + idx()*size;
	for(int i = 0; i < size; i++) 
		vec[i] = 0;
}

__device__ inline float get_e(int i)
{
	if(i == 0 || i == 1) return 1;
	else return 2;
}

__device__ inline float get_D(int i, int j, float E)
{
	float ni = 0.3;
	float mi = E/(2*(1+ni));
	float lambda = ni*E/(1+ni)/(1-2*ni);
	
	if(i == 2 && j == 2) return mi;
	else if(i > 1 || j > 1) return 0;
	else if(i == j) return lambda + 2*mi;
	else return lambda;
}

__device__ inline float get_alpha(int i)
{
	float base = -0.06115;
	if(i == 0) return base;
	if(i == 1) return 0;
	return 2*base;
}

__global__ void fillInside(matrix* insideMatrices, float E1, float E2, int size){
	matrix* myMatrix = insideMatrices+(idx()/(2*size));
	int myRow = idx()%(2*size);
	
	if (myRow == 0 || myRow == size-1){
		myMatrix->ul[myRow*size+myRow] = 0.5;
		myMatrix->ub[myRow] = 10.0/size*(idx()/(2*size)+1);
	}else if (myRow == size || myRow == 2*size-1){
		myRow -= size;
		myMatrix->lr[myRow*size+myRow] = 0.5;
		myMatrix->lb[myRow] = 10.0/size*(idx()/(2*size)+2);
	}else if(myRow < size){
		myMatrix->ul[myRow*size+myRow] = -2.0;
		myMatrix->ul[myRow*size+myRow-1] = (float)1/2;
		myMatrix->ul[myRow*size+myRow+1] = (float)1/2;
		myMatrix->ur[myRow*size+myRow] = 1;
	}else{
		myRow -= size;
		myMatrix->lr[myRow*size+myRow] = -2.0;
		myMatrix->lr[myRow*size+myRow-1] = (float)1/2;
		myMatrix->lr[myRow*size+myRow+1] = (float)1/2;
		myMatrix->ll[myRow*size+myRow] = 1;
	}
}

__global__ void fillLeft(matrix* leftMatrix, float E, int size){
	int myRow = idx()%size;
	
	leftMatrix->ul[myRow*size+myRow] = 1;
	leftMatrix->ub[myRow] = 0;

	float a = 0;
	float right = 0;
	for(int i = 0; i < 3; i++)
		for(int j = 0; j < 3; j++)
		{
			a += get_e(i)*get_e(j)*get_D(i,j,E);
			right += get_e(i)*get_D(i,j,E)*get_alpha(j);
		}

	leftMatrix->ll[myRow*size + myRow] = a;
	leftMatrix->lr[myRow*size + myRow] = a/2;
	if (myRow >= 2) leftMatrix->lr[myRow*size+myRow-2] = a/2;
	if (myRow < size-2) leftMatrix->lr[myRow*size+myRow+2] = a/2;
	leftMatrix->lb[myRow] = right;
}

__global__ void fillRight(matrix* rightMatrix, int size){
	int myRow = idx()%size;
	
	leftMatrix->lr[myRow*size+myRow] = 1;
	leftMatrix->lb[myRow] = 0;

	float a = 0;
	float right = 0;
	for(int i = 0; i < 3; i++)
		for(int j = 0; j < 3; j++)
		{
			a += get_e(i)*get_e(j)*get_D(i,j,E);
			right += get_e(i)*get_D(i,j,E)*get_alpha(j);
		}

	//No boundary condition so these are not connected to the last row, which is just throwaway
	leftMatrix->ul[myRow*size + myRow] = a/2;
	if (myRow >= 2) leftMatrix->ul[myRow*size+myRow-2] = a/2;
	if (myRow < size-2) leftMatrix->ul[myRow*size+myRow+2] = a/2;
	leftMatrix->lb[myRow] = right;
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

__global__ void calculateElement(float* dElement, int row, matrix* A, int size){
	A += idx()*2;
	dElement[idx()] = A->lr[row*(size+1)] + (A+1)->ul[row*(size+1)];
}

__global__ void countCoeffsUpper(float* dElement, matrix* A, int col, float* coeffs, int size) {
	coeffs[idx()] = A->ur[idx()*size + col] / (*dElement);
}

__global__ void countCoeffsLower(float* dElement, matrix* B, int col, float* coeffs, int size) {
	coeffs[idx()] = B->ll[idx()*size + col] / (*dElement);
}

__global__ void countCoeffsCenter(float* dElement, matrix* A, matrix* B, int col, float* coeffs, int size) {
	int newIdx = idx() + col + 1;
	int index = newIdx*size + col;

	coeffs[newIdx] = (A->lr[index] + B->ul[index]) / (*dElement);
}

__global__ void updateUpperLeft(int row, matrix* A, matrix* C, float* coeffs, int size) {
	int x = idx() % size;
	int y = idx() / size;
	
	C->ul[idx()] = C->ul[idx()] - A->ll[row*size + x]*coeffs[y];
}

__global__ void updateLowerRight(int row, matrix* B, matrix* C, float* coeffs, int size) {
	int x = idx() % size;
	int y = idx() / size;
	
	C->lr[idx()] = C->lr[idx()] - B->ur[row*size + x]*coeffs[y];
}

__global__ void updateLowerLeft(int row, matrix* A, matrix* C, float* coeffs, int size) {
	int x = idx() % size;
	int y = idx() / size;
	
	C->ll[idx()] = C->ll[idx()] - A->ll[row*size + x] * coeffs[y];
}

__global__ void updateUpperRight(int row, matrix* B, matrix* C, float* coeffs, int size) {
	int x = idx() % size;
	int y = idx() / size;
	
	C->ur[idx()] = C->ur[idx()] - B->ur[row*size + x] * coeffs[y];
}

__global__ void updateUpperCenter(int row, matrix* A, matrix* B, float* coeffs, int size) {
	int x = idx() % size;
	int y = idx() / size;

	int index = (row * size) + x;
	A->ur[idx()] = A->ur[idx()] - (A->lr[index] + B->ul[index]) * coeffs[y];
}

__global__ void updateLowerCenter(int row, matrix* A, matrix* B, float* coeffs, int size) {
	int x = idx() % size;
	int y = idx() / size;

	int index = (row * size) + x;
	B->ll[idx()] = B->ll[idx()] - (A->lr[index] + B->ul[index]) * coeffs[y];
}

__global__ void updateCenter(int row, matrix* A, matrix* B, float* coeffs, int size) {
	int newIdx = idx() + (row + 1) * size;
	int x = newIdx % size;
	int y = newIdx / size;
	
	int index = (row * size) + x;
	float element = A->lr[newIdx] + B->ul[newIdx] - (A->lr[index] + B->ul[index]) * coeffs[y];
	
	A->lr[newIdx] = element / 2;
	B->ul[newIdx] = element / 2;
}

__global__ void updateLeftCenter(int row, matrix* A, float* coeffs, int size) {
	int newIdx = idx() + (row + 1) * size;
	int x = newIdx % size;
	int y = newIdx / size;
	
	A->ll[newIdx] = A->ll[newIdx] - A->ll[(row * size) + x] * coeffs[y];
}

__global__ void updateRightCenter(int row, matrix* B, float* coeffs, int size) {
	int newIdx = idx() + (row + 1) * size;
	int x = newIdx % size;
	int y = newIdx / size;

	B->ur[newIdx] = B->ur[newIdx] - B->ur[(row * size) + x] * coeffs[y];
}

__global__ void updateBCenter(int row, matrix* A, matrix* B, float* coeffs, int size) {
	int newIdx = row + 1 + idx();
	float element = A->lb[newIdx] + B->ub[newIdx] - (A->lb[row] + B->ub[row])*coeffs[newIdx];
	A->lb[newIdx] = element / 2;
	B->ub[newIdx] = element / 2;
}

__global__ void updateBLower(int row, matrix* A, matrix* B, matrix* C, float* coeffs, int size) {
	int newIdx = idx();
	C->lb[newIdx] = C->lb[newIdx] - (B->ub[row] + A->lb[row]) * coeffs[newIdx];
}

__global__ void updateBUpper(int row, matrix* A, matrix* B, matrix* C, float* coeffs, int size) {
	int newIdx = idx();
	
	C->ub[newIdx] = C->ub[newIdx] - (B->ub[row] + A->lb[row]) * coeffs[newIdx];
}

__global__ void backwardsSubstitutionRight(int col, matrix* A, matrix* B, int size) {
	//No need to divide here - these values came from the sky
	//To use this when there is no boundary condition on one edge
	//we need en extra row of variables set to 0 and unrelated to the rest
	int index = idx();
	float element = A->lb[index] + B->ub[index] - B->ur[index*size + col] * B->lb[col];
	A->lb[index] = element/2;
	B->ub[index] = element/2;
}

__global__ void backwardsSubstitutionLeft(int col, matrix* A, matrix* B, int size) {
	int index = idx();
	float element = A->lb[index] + B->ub[index] - A->ll[index*size + col] * A->ub[col];
	A->lb[index] = element/2;
	B->ub[index] = element/2;
}

__global__ void backwardsSubstitutionCenter(int col, matrix* A, matrix* B, int size) {
	int index = idx();
	int index2d = index*size + col;
	int indexCoeff = col*size + col;
	//So far we're keeping the entries undivided, so a division is needed
	float element = A->lb[index]+B->ub[index] - (A->lb[col]+B->ub[col])*(A->lr[index2d]+B->ul[index2d])/(A->lr[indexCoeff]+B->ul[indexCoeff]);
	A->lb[index] = element/2;
	B->ub[index] = element/2;
}

__global__ void extractResults(matrix* A, float* target, int size) {
	A += idx()*2;
	matrix* B = A + 1;
	target += idx()*size*2;
	for(int i = 0; i < size; i++)
	{
		target[i] = A->ub[i];
		target[i+size] = (A->lb[i] + B->ub[i])/(A->lr[i*(size+1)] + B->ul[i*(size+1)]);
		target[i+2*size] = B->lb[i];
	}
}

__global__ void copyBLower(matrix* A, matrix* B, matrix* C, int size) {
	int index = idx();
	C->lb[index] = (A->lb[index] + B->ub[index])/(A->lr[index*(size+1)] + B->ul[index*(size+1)]);
}

__global__ void copyBUpper(matrix* A, matrix* B, matrix* C, int size) {
	int index = idx();
	C->ub[index] = (A->lb[index] + B->ub[index])/(A->lr[index*(size+1)] + B->ul[index*(size+1)]);
}

void printDeviceMatrix(matrix* deviceMatrix, int size){
	matrix temp;
	float* temp_data = (float*)malloc(matrix_size(size) * sizeof(float));
	cudaMemcpy(&temp, deviceMatrix, sizeof(matrix), cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();	
	cudaMemcpy(temp_data, temp.ur, sizeof(float) * matrix_size(size), cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
	
	temp.ur = temp_data;
	temp.ul = temp.ur + size*size;
	temp.lr = temp.ul + size*size;
	temp.ll = temp.lr + size*size;
	temp.ub = temp.ll + size*size;
	temp.lb = temp.ub + size;

	printHostMatrix(&temp, size);
	free(temp_data);
}

void printHostMatrix(matrix* temp, int size) {
	printf("ul\n");
	printFloatArray(temp->ul, size);
	printf("ur\n");
	printFloatArray(temp->ur, size);
	printf("ll\n");
	printFloatArray(temp->ll, size);
	printf("lr\n");
	printFloatArray(temp->lr, size);
	printf("ub\n");
	printHostVector(temp->ub, size);
	printf("lb\n");
	printHostVector(temp->lb, size);
	printf("\n");
}

void printFloatArray(float *M, int size) {
	int i,j;
	for(i = 0; i < size; i++){
		for(j = 0; j < size; j++) {
			printf("% .2f ", M[i*size + j]);
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
		printf("% .2f ", V[i]);
	printf("\n\n");
}