#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "solver.h"

void printMatrix(matrix* m, int size);

float get(matrix* m, int x, int y, int size) {
	if (x == 2*size && y < size)
		return m-> ub[y];
	if (x == 2*size && y >= size)
		return m-> lb[y-size];
		
	if (x < size && y < size)
		return m-> ul[y*size + x];
	if (x < size && y >= size)
		return m-> ll[(y-size)*size + x];
	if (x >= size && y < size)
		return m-> ur[y*size + (x-size)];
	if (x >= size && y >= size)
		return m-> lr[(y-size)*size + (x-size)];
		
	return -1;
}

void set(matrix* m, int x, int y, int size, float el) {
	if (x == 2*size && y < size)
		m-> ub[y] = el;
	else if (x == 2*size && y >= size)
		m-> lb[y-size] = el;
	else if (x < size && y < size)
		m-> ul[y*size + x] = el;
	else if (x < size && y >= size)
		m-> ll[(y-size)*size + x] = el;
	else if (x >= size && y < size)
		m-> ur[y*size + (x-size)] = el;
	else if (x >= size && y >= size)
		m-> lr[(y-size)*size + (x-size)] = el;	
}

void gauss(matrix* m, int size) {
	for(int i = 0; i < size * 2; i++) {
		for(int j = i + 1; j < size * 2; j++) {
			float coeff = get(m, i, j, size) / get(m, i, i, size);
			for(int k = 0; k < size * 2 + 1; k++)
				set(m, k, j, size, get(m, k, j, size) - coeff * get(m, k, i, size));
		}
	}
	
	printMatrix(m, size);
	
	for(int i = size*2 - 1; i >= 0; i--) {
		set(m, 2*size, i, size, get(m, 2*size, i,size) / get(m, i, i, size));
		set(m, i, i, size, 1.0f);
		for(int j = i - 1; j >= 0; j--) {
			set(m, 2*size, j, size, get(m, 2*size, j, size) - get(m, i, j, size) * get(m,2*size, i, size));
			set(m, i, j, size, 0.0f);
		}
	}
}

matrix* createMatrix(int size){
	matrix* temp = (matrix*)malloc(sizeof(matrix));
	float* temp_data = (float*)malloc(matrix_size(size) * sizeof(float));
	
	temp->ur = temp_data;
	temp->ul = temp->ur + size*size;
	temp->lr = temp->ul + size*size;
	temp->ll = temp->lr + size*size;
	temp->ub = temp->ll + size*size;
	temp->lb = temp->ub + size;

	return temp;
}

void printMatrix(matrix* m, int size) {
	for(int y = 0; y < size*2; y++) {
		for(int x = 0; x < size*2 + 1; x++)
			printf("%f ", get(m, x, y, size));
		printf("\n");
	}
	printf("\n");
}

void setMatrix(matrix* m, int size) {
	srand(27);
	for(int y = 0; y < size*2; y++) 
		for(int x = 0; x < size*2 + 1; x++)
			set(m, x, y, size, rand() % 100);
		
}

int main(char** args) {
	int size = 2;
	matrix* m = createMatrix(size);
	setMatrix(m, size);
	printMatrix(m, size);
	
	gauss(m, size);
	printMatrix(m, size);
}
