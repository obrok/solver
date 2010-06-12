#include "solver.h"

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
	
	for(int i = size*2 - 1; i >= 0; i--) {
		set(m, 2*size, i, size, get(m, 2*size, i,size) / get(m, i, i, size));
		set(m, i, i, size, 1.0f);
		for(int j = i - 1; j >= 0; j--) {
			set(m, 2*size, j, size, get(m, 2*size, j, size) - get(m, i, j, size) * get(m,2*size, i, size));
			set(m, i, j, size, 0.0f);
		}
	}
}
