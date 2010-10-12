#include <stdio.h>
#include <cuda.h>
#include <windows.h>
#include <iostream>

#include "solver.h"
#include "gauss.h"

void reduce(matrix* oldMatrices, int n, matrix* newMatrices, int size);
void solve(matrix* matrices, int n, int size);
int calculate(int _size, int _log);

int main()
{
	calculate(4, 2);		
}

void allocate_tree(matrix** matrices, float** datas, int matrix_no, int size, int log)
{	
	cudaMalloc((void**)&(matrices[0]), sizeof(matrix)*matrix_no);
	cudaMalloc((void**)&(datas[0]), matrix_no*sizeof(float)*matrix_size(size));
	cudaThreadSynchronize();
	
	init_matrices<<<matrix_no, 1>>>(matrices[0], datas[0], size);
	cudaThreadSynchronize();
	
	for(int i = 0, n = matrix_no; i < log; i++,n/=2)
	{
		cudaMalloc((void**)&matrices[i+1], sizeof(matrix)*n/2);
		cudaMalloc((void**)&datas[i+1], sizeof(float)*matrix_size(size)*n/2);
		cudaThreadSynchronize();
		
		init_matrices<<<n/2, 1>>>(matrices[i+1], datas[i + 1], size);
		cudaThreadSynchronize();
	}
}

void free_tree(matrix** matrices, float** datas, int matrix_no, int size, int log)
{	
	for(int i = 0, n = matrix_no; i < log; i++,n/=2)
	{
		cudaFree(matrices[i]);
		cudaFree(datas[i]);
		cudaThreadSynchronize();
	}
	cudaThreadSynchronize();
}

void fill_tree(matrix* bottom_level, int matrix_no, int size, bool left_open, bool right_open)
{
	if(left_open)
	{
		fillInside<<<1, 2*size>>>(bottom_level,  size);
	}
	else
	{
		fillLeft<<<1, 2*size>>>(bottom_level,  size);		
	}
	fillInside<<<matrix_no-2, 2*size>>>(bottom_level + 1, size);
	if(right_open)
	{
		fillInside<<<1, 2*size>>>(bottom_level+matrix_no-1, size);
	}
	else
	{
		fillRight<<<1, 2*size>>>(bottom_level+matrix_no-1, size);
	}
	cudaThreadSynchronize();
}

void solve_up(matrix** matrices, int matrix_no, int size, int log)
{
	int i;
	int n ;		
	for(i = 0,n=matrix_no; i < log; i++,n/=2){
		reduce(matrices[i], n, matrices[i+1], size);			
	}
}

void solve_top(matrix** matrices, int matrix_no, int size, int log)
{
	matrix temp;
	float* temp_data = (float*)malloc(matrix_size(size) * sizeof(float));
	cudaMemcpy(&temp, matrices[log], sizeof(matrix), cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
	cudaMemcpy(temp_data, temp.ur, sizeof(float) * matrix_size(size), cudaMemcpyDeviceToHost);
	float* target = temp.ur;
	cudaThreadSynchronize();
	
	temp.ur = temp_data;
	temp.ul = temp.ur + size*size;
	temp.lr = temp.ul + size*size;
	temp.ll = temp.lr + size*size;
	temp.ub = temp.ll + size*size;
	temp.lb = temp.ub + size;
	
	gauss(&temp, size);
	
	cudaMemcpy(target, temp_data, sizeof(float) * matrix_size(size), cudaMemcpyHostToDevice);
	cudaThreadSynchronize();
	
	matrix top_left, top_right;
	cudaMemcpy(&top_left, matrices[log-1], sizeof(matrix), cudaMemcpyDeviceToHost);
	cudaMemcpy(&top_right, matrices[log-1]+1, sizeof(matrix), cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
	cudaMemcpy(top_left.ul, temp.ul, sizeof(float) * size * size, cudaMemcpyHostToDevice);
	cudaMemcpy(top_left.ur, temp.ur, sizeof(float) * size * size, cudaMemcpyHostToDevice);	
	cudaMemcpy(top_left.ub, temp.ub, sizeof(float) * size, cudaMemcpyHostToDevice);
	cudaMemcpy(top_right.ll, temp.ll, sizeof(float) * size * size, cudaMemcpyHostToDevice);
	cudaMemcpy(top_right.lr, temp.lr, sizeof(float) * size * size, cudaMemcpyHostToDevice);	
	cudaMemcpy(top_right.lb, temp.lb, sizeof(float) * size, cudaMemcpyHostToDevice);
	cudaThreadSynchronize();
	
	free(temp_data);
}

void solve_down(matrix** matrices, int matrix_no, int size, int log)
{			
	for(int i = log-1,n=2; i >= 0; i--,n*=2){
		solve(matrices[i], n, size);

		if(i > 0)
			for(int j = 0; j < n; j+=2){
				copyBUpper<<<1, size>>>(matrices[i]+j, matrices[i-1]+j*2);
				copyBLower<<<1, size>>>(matrices[i]+j, matrices[i]+j+1, matrices[i-1]+j*2+1, size);
				copyBUpper<<<1, size>>>(matrices[i]+j, matrices[i]+j+1, matrices[i-1]+j*2+2, size);
				copyBLower<<<1, size>>>(matrices[i]+j+1, matrices[i-1]+j*2+3);								
			}
				
		cudaThreadSynchronize();
	}
}

int calculate(int _size, int _log){
	int size = _size;
	int log = _log;
	int matrix_no = size;
	size = size + 1;
	
	log /= 2;
	for(int i = 0; i < log; i++)
	{
		matrix_no /= 2;
	}
	
	float* carry;
	cudaMalloc((void**)&carry, matrix_no*sizeof(float)*matrix_size(size));
	
	matrix** matrices = (matrix**)malloc(sizeof(matrix*)*(log + 1));
	float** data = (float**)malloc(sizeof(float*)*(log + 1));
	
	for(int i = 0; i < matrix_no; i++)
	{
		allocate_tree(matrices, data, matrix_no, size, log);
	
		fill_tree(matrices[0], matrix_no, size, i != 0, i != matrix_no - 1);
	
		solve_up(matrices, matrix_no, size, log);
		
		cudaMemcpy(carry + i*matrix_size(size), data[log], sizeof(float) * matrix_size(size), cudaMemcpyDeviceToDevice);
		cudaThreadSynchronize();
		
		free_tree(matrices, data, matrix_no, size, log);
	}
	
	printDeviceVector(carry, matrix_no*matrix_size(size));
	
	allocate_tree(matrices, data, matrix_no, size, log);
	
	cudaMemcpy(data[0], carry, sizeof(float) * matrix_no * matrix_size(size), cudaMemcpyDeviceToDevice);
	cudaThreadSynchronize();
	
	printDeviceMatrix(matrices[0], size);
	printDeviceMatrix(matrices[0]+1, size);
	
	solve_up(matrices, matrix_no, size, log);
	
	solve_top(matrices, matrix_no, size, log);
	
	solve_down(matrices, matrix_no, size, log);

	cudaMemcpy(carry, data[0], sizeof(float) * matrix_no * matrix_size(size), cudaMemcpyDeviceToDevice);
	cudaThreadSynchronize();
	
	free_tree(matrices, data, matrix_no, size, log);
	
	for(int i = 0; i < matrix_no; i++)
	{
		allocate_tree(matrices, data, matrix_no, size, log);
	
		fill_tree(matrices[0], matrix_no, size, i != 0, i != matrix_no - 1);
	
		solve_up(matrices, matrix_no, size, log);
		
		cudaMemcpy(data[log], carry + i*matrix_size(size), sizeof(float) * matrix_size(size), cudaMemcpyDeviceToDevice);
		cudaThreadSynchronize();
		
		float* values;
		cudaMalloc((void**)&values, sizeof(float)*(matrix_no+1)*size);
		cudaThreadSynchronize();
		extractResults<<<1, matrix_no/2>>>(matrices[0], values, size);
		cudaThreadSynchronize();
	
		printDeviceVector(values, (matrix_no+1)*size);
		
		free_tree(matrices, data, matrix_no, size, log);
	}
	float* values;
	cudaMalloc((void**)&values, sizeof(float)*(matrix_no+1)*size);
	cudaThreadSynchronize();
	extractResults<<<1, matrix_no/2>>>(matrices[0], values, size);
	cudaThreadSynchronize();
	
	free_tree(matrices, data, matrix_no, size, log);
	
	return 0;
}

void reduce(matrix* oldMatrices, int n, matrix* newMatrices, int size){
	int i;
	
	for(i = 0; i < n; i += 2){
		copyUpperLeft<<<size,size>>>(oldMatrices+i, newMatrices+i/2);
		copyLowerRight<<<size,size>>>(oldMatrices+i+1, newMatrices+i/2);
		copyBUpper<<<1,size>>>(oldMatrices+i, newMatrices+i/2);
		copyBLower<<<1,size>>>(oldMatrices+i+1, newMatrices+i/2);
	}
	
	float* coeffs;
	cudaMalloc((void**)&coeffs, sizeof(float)*3*size*n);
	cudaThreadSynchronize();
	init_vector<<<3*n, 1>>>(coeffs, size);
	cudaThreadSynchronize();
		
	int row;
	for(row = 0; row < size; row++){
		float* dElement;
		cudaMalloc((void**)&dElement, sizeof(float)*n/2);
		cudaThreadSynchronize();
		calculateElement<<<1,n/2>>>(dElement, row, oldMatrices, size);
		cudaThreadSynchronize();
		
		//Calculate coefficients
		for(i = 0; i < n; i += 2){
			countCoeffsUpper<<<1, size>>>(dElement+i/2, oldMatrices+i, row, coeffs+(3*size)*i, size);
			countCoeffsCenter<<<1,size-row-1>>>(dElement+i/2, oldMatrices+i, oldMatrices+i+1, row, coeffs+(3*size)*i+size, size);
			countCoeffsLower<<<1, size>>>(dElement+i/2, oldMatrices+i+1, row, coeffs+(3*size)*i+2*size, size);
		}
		cudaThreadSynchronize();
		
		for(i = 0; i < n; i += 2){
			updateUpperLeft<<<size, size>>>(row, oldMatrices+i, newMatrices+i/2, coeffs+(3*size)*i,  size);
			updateUpperCenter<<<size, size>>>(row, oldMatrices+i, oldMatrices+i+1, coeffs+(3*size)*i,  size);
			updateUpperRight<<<size, size>>>(row, oldMatrices+i+1, newMatrices+i/2, coeffs+(3*size)*i,  size);
			updateBUpper<<<1, size>>>(row, oldMatrices+i, oldMatrices+i+1, newMatrices+i/2, coeffs+(3*size)*i,  size);
			
			updateLeftCenter<<<size-row-1, size>>>(row, oldMatrices+i, coeffs+(3*size)*i+size,  size);
			updateCenter<<<size-row-1, size>>>(row, oldMatrices+i, oldMatrices+i+1, coeffs+(3*size)*i+size,  size);
			updateRightCenter<<<size-row-1, size>>>(row, oldMatrices+i+1, coeffs+(3*size)*i+size,  size);
			updateBCenter<<<1, size-row-1>>>(row, oldMatrices+i, oldMatrices+i+1, coeffs+(3*size)*i+size,  size);
			
			updateLowerLeft<<<size, size>>>(row, oldMatrices+i, newMatrices+i/2, coeffs+(3*size)*i+2*size,  size);
			updateLowerCenter<<<size, size>>>(row, oldMatrices+i, oldMatrices+i+1, coeffs+(3*size)*i+2*size,  size);
			updateLowerRight<<<size, size>>>(row, oldMatrices+i+1, newMatrices+i/2, coeffs+(3*size)*i+2*size,  size);
			updateBLower<<<1, size>>>(row, oldMatrices+i, oldMatrices+i+1, newMatrices+i/2, coeffs+(3*size)*i+2*size,  size);
			
		}
		
		cudaThreadSynchronize();
	}
	
	cudaFree(coeffs);
}

void solve(matrix* matrices, int n, int size){
	int i, j;
	
	for(i = 0; i < size; i++){
		for(j = 0; j < n; j+=2)
			backwardsSubstitutionRight<<<1, size>>>(i, matrices+j, matrices+j+1, size);
		cudaThreadSynchronize();
		for(j = 0; j < n; j+=2)
			backwardsSubstitutionLeft<<<1,size>>>(i, matrices+j, matrices+j+1, size);
		cudaThreadSynchronize();
	}
	
	for(i = 0; i < size; i++){
		for(j = 0; j < n; j+=2)
			backwardsSubstitutionCenter<<<1, size-i-1>>>(size-i-1, matrices+j, matrices+j+1, size);
		cudaThreadSynchronize();
	}
	
}

void fix_top(matrix* matrix, int size)
{
}