#include <cuda.h>

typedef struct {
	float* ur; // [N*N];
	float* ul; //[N*N];
	float* lr; //[N*N];
	float* ll; //[N*N];
	float* ub; //[N];
	float* lb; //[N];
} matrix;


__host__ __device__ int matrix_size(int size);

__device__ int idx();

__global__ void calculateElement(float* dElement, int row, matrix* A, int size);

__global__ void copyUpperLeft(matrix* A, matrix* C);
__global__ void copyLowerRight(matrix* B, matrix* C);

__global__ void copyBUpper(matrix* A, matrix* C);
__global__ void copyBLower(matrix* B, matrix* C);

__global__ void fillInside(matrix* insideMatrices, int size);
__global__ void fillLeft(matrix* leftMatrix, float E1, int size);
__global__ void fillRight(matrix* rightMatrix, int size);

__global__ void countCoeffsUpper(float* dElement, matrix* A, int col, float* coeffs, int size);
__global__ void countCoeffsLower(float* dElement, matrix* B, int col, float* coeffs, int size);
__global__ void countCoeffsCenter(float* dElement, matrix* A, matrix* B, int col, float* coeffs, int size);

__global__ void updateUpperLeft(int row, matrix* A, matrix* C, float* coeffs, int size);
__global__ void updateLowerRight(int row, matrix* B, matrix* C, float* coeffs, int size);
__global__ void updateLowerLeft(int row, matrix* A, matrix* C, float* coeffs, int size);
__global__ void updateUpperRight(int row, matrix* B, matrix* C, float* coeffs, int size);

__global__ void updateUpperCenter(int row, matrix* A, matrix* B, float* coeffs, int size);
__global__ void updateLowerCenter(int row, matrix* A, matrix* B, float* coeffs, int size);

__global__ void updateCenter(int row, matrix* A, matrix* B, float* coeffs, int size);
__global__ void updateLeftCenter(int row, matrix* A, float* coeffs, int size);
__global__ void updateRightCenter(int row, matrix* B, float* coeffs, int size);

__global__ void updateBCenter(int row, matrix* A, matrix* B, float* coeffs, int size);
__global__ void updateBLower(int row, matrix* A, matrix* B, matrix* C, float* coeffs, int size);
__global__ void updateBUpper(int row, matrix* A, matrix* B, matrix* C, float* coeffs, int size);

__global__ void backwardsSubstitutionRight(int col, matrix* A, matrix* B, int size);
__global__ void backwardsSubstitutionLeft(int col, matrix* A, matrix* B, int size);
__global__ void backwardsSubstitutionCenter(int col, matrix* A, matrix* B, int size);

__global__ void copyBLower(matrix* A, matrix* B, matrix* C, int size);
__global__ void copyBUpper(matrix* A, matrix* B, matrix* C, int size);
__global__ void extractResults(matrix* A, float* target, int size);

__global__ void init_matrices(matrix* A, float* data, int size);
__global__ void init_vector(float* vec, int size);

void printDeviceMatrix(matrix* deviceMatrix, int size);
void printHostMatrix(matrix* hostMatrix, int size);
void printDeviceVector(float* dVec, int len);
void printHostVector(float* V, int len);
void printFloatArray(float *M, int size);