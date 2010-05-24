#include <stdio.h>
#include <cuda.h>
#include "CuTest.h"
#include "Solver.h"

float* deviceMallocElementAndFill(float val) {
	float* dResult;
	cudaMalloc((void**)&dResult, sizeof(float));
	cudaMemcpy(dResult, &val, sizeof(float), cudaMemcpyHostToDevice);
	cudaThreadSynchronize();
	return dResult;
}

matrix* deviceMallocMatrixAndZero() {
	matrix* hA = (matrix*)malloc(sizeof(matrix));
	
	int i;
	for(i = 0; i < N*N; i++) {
		hA->ul[i] = 0;
		hA->ur[i] = 0;
		hA->ll[i] = 0;
		hA->lr[i] = 0;
	}
	
	for(int i = 0; i < N; i++) {
		hA->ub[i] = 0;
		hA->lb[i] = 0;
	}
	
	matrix* dA;
	cudaMalloc((void**)&dA, sizeof(matrix));
	cudaMemcpy(dA, hA, sizeof(matrix), cudaMemcpyHostToDevice);
	cudaThreadSynchronize();
	
	free(hA);
	return dA;
}

matrix* deviceMallocMatrixAndFill() {
	matrix* hA = (matrix*)malloc(sizeof(matrix));
	
	int i;
	for(i = 0; i < N*N; i++) {
		hA->ul[i] = i;
		hA->ur[i] = i + N*N;
		hA->ll[i] = i + 2*N*N;
		hA->lr[i] = i + 3*N*N;
	}
	
	for(int i = 0; i < N; i++) {
		hA->ub[i] = i;
		hA->lb[i] = i + N;
	}
	
	matrix* dA;
	cudaMalloc((void**)&dA, sizeof(matrix));
	cudaMemcpy(dA, hA, sizeof(matrix), cudaMemcpyHostToDevice);
	cudaThreadSynchronize();
	
	free(hA);
	return dA;
}

float* deviceMallocCoeffsAndFill(float val) {
	float* hCoeffs = (float*)malloc(sizeof(float)*N);
	
	int i;
	for(i = 0; i < N; i++) {
		hCoeffs[i] = val;
	}
	
	float* dCoeffs;
	
	cudaMalloc((void**)&dCoeffs, sizeof(float) * N);
	cudaMemcpy(dCoeffs, hCoeffs, sizeof(float) * N, cudaMemcpyHostToDevice);
	cudaThreadSynchronize();
	
	free(hCoeffs);
	return dCoeffs;
}

float* deviceMallocCoeffsAndZero() {
	return deviceMallocCoeffsAndFill(0.0f);
}

void prepareDataForUpdateTest(matrix** dA, matrix** dB, matrix** dC) {
	*dA = deviceMallocMatrixAndFill();
	*dB = deviceMallocMatrixAndFill();
	*dC = deviceMallocMatrixAndZero();
	

	int i;
	for(i = 0; i < N; i += 2){
		copyUpperLeft<<<N,N>>>(*dA, *dC);
		copyLowerRight<<<N,N>>>(*dB, *dC);
		copyBUpper<<<1,N>>>(*dA, *dC);
		copyBLower<<<1,N>>>(*dB, *dC);
	}
	
	cudaThreadSynchronize();
}

matrix* downloadMatrix(matrix* device) {
	matrix* result = (matrix*)malloc(sizeof(matrix));
	
	cudaMemcpy(result, device, sizeof(matrix), cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
	
	return result;
}

float* downloadCoeffs(float* device) {
	float* result = (float*)malloc(sizeof(float)*N);
	
	cudaMemcpy(result, device, sizeof(float)*N, cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
	
	return result;
}

float downloadElement(float* element) {
	float result;
	cudaMemcpy(&result, element, sizeof(float), cudaMemcpyDeviceToHost);
	return result;
}

void TestPrepareDataForUpdateTest(CuTest *tc) {
	matrix* dA = NULL;
	matrix* dB = NULL;
	matrix* dC = NULL;
	prepareDataForUpdateTest(&dA, &dB, &dC);

	matrix* hC = downloadMatrix(dC);
	
	double actual = hC->ul[1];
	double expected = 1;
	CuAssertDblEquals(tc, expected, actual, 0.001);
	
	actual = hC->ur[1];
	expected = 0;
	CuAssertDblEquals(tc, expected, actual, 0.001);
	
	actual = hC->ll[1];
	expected = 0;
	CuAssertDblEquals(tc, expected, actual, 0.001);
	
	actual = hC->lr[1];
	expected = 49;
	CuAssertDblEquals(tc, expected, actual, 0.001);
	
	free(hC);
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
}

void TestCountUpper(CuTest *tc) {
	matrix* dA = deviceMallocMatrixAndFill();
	float* dCoeffs = deviceMallocCoeffsAndZero();
	float* dElement = deviceMallocElementAndFill(2.0f);
	
	countCoeffsUpper<<<1, N>>>(dElement, dA, 0, dCoeffs);
	cudaThreadSynchronize();

	float* hCoeffs = downloadCoeffs(dCoeffs);
		
	double actual = hCoeffs[0];
    double expected = 8.0;
	CuAssertDblEquals(tc, expected, actual, 0.001);
	
	actual = hCoeffs[3];
	expected = 14.0;
	CuAssertDblEquals(tc, expected, actual, 0.001);
	
	free(hCoeffs);
	cudaFree(dElement);
	cudaFree(dA);
	cudaFree(dCoeffs);
}

void TestCountLower(CuTest *tc) {
	matrix* dB = deviceMallocMatrixAndFill();
	float* dCoeffs = deviceMallocCoeffsAndZero();
	float* dElement = deviceMallocElementAndFill(2.0f);
	
	countCoeffsLower<<<1, N>>>(dElement, dB, 0, dCoeffs);
	cudaThreadSynchronize();
	
	float* hCoeffs = downloadCoeffs(dCoeffs);
	
	double actual = hCoeffs[0];
    double expected = 16;
	CuAssertDblEquals(tc, expected, actual, 0.001);
	
	actual = hCoeffs[3];
	expected = 22;
	CuAssertDblEquals(tc, expected, actual, 0.001);
	
	free(hCoeffs);
	cudaFree(dElement);
	cudaFree(dB);
	cudaFree(dCoeffs);
}

void TestCountCenter(CuTest *tc) {
	matrix* dA = deviceMallocMatrixAndFill();
	matrix* dB = deviceMallocMatrixAndFill();
	float* dCoeffs = deviceMallocCoeffsAndZero();
	float* dElement = deviceMallocElementAndFill(2.0f);
	
	countCoeffsCenter<<<1, N - 1>>>(dElement, dA, dB, 0, dCoeffs);
	cudaThreadSynchronize();
	
	float* hCoeffs = downloadCoeffs(dCoeffs);
		
	double actual = hCoeffs[1];
    double expected = 28;
    
	CuAssertDblEquals(tc, expected, actual, 0.001);
	
	if (N >= 3) {
		actual = hCoeffs[3];
		expected = 36;

		CuAssertDblEquals(tc, expected, actual, 0.001);
	}
	
	free(hCoeffs);
	cudaFree(dElement);
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dCoeffs);
}

void TestCalculateElement(CuTest *tc) {
	matrix* dA = deviceMallocMatrixAndFill();
	matrix* dB = deviceMallocMatrixAndFill();
	
	float* dElement;
	cudaMalloc((void**)&dElement, sizeof(float));
	
	calculateElement<<<1, 1>>>(dElement, 0, dA, dB);
	cudaThreadSynchronize();
	
	double actual = downloadElement(dElement);
	double expected = 48.0f;
	
	CuAssertDblEquals(tc, expected, actual, 0.001);
	
	cudaFree(dA);
	cudaFree(dB);
}

void TestUpdateUpperLeft(CuTest *tc) {
	matrix* dA;
	matrix* dB;
	matrix* dC;
	prepareDataForUpdateTest(&dA, &dB, &dC);
	
	float* dCoeffs = deviceMallocCoeffsAndZero();
	
	float* element;
	cudaMalloc((void**)&element, sizeof(float));
	
	calculateElement<<<1,1>>>(element, 0, dA, dB);
	cudaThreadSynchronize();
	
	countCoeffsUpper<<<1,N>>>(element, dA, 0, dCoeffs);
	cudaThreadSynchronize();
	
	updateUpperLeft<<<1,N*N>>>(0, dA, dC, dCoeffs);
	cudaThreadSynchronize();
	
	matrix* hC = downloadMatrix(dC);
	
	double actual = hC->ul[0];
	double expected = - (32 * 16) / 48.0f;
	CuAssertDblEquals(tc, expected, actual, 0.0001);
	
	actual = hC->ul[15];
	expected = 15 - (28*35) /48.0f;
	CuAssertDblEquals(tc, expected, actual, 0.0001);
	
	free(hC);
	cudaFree(element);
	cudaFree(dCoeffs);
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
}

void TestUpdateLowerLeft(CuTest *tc) {
	matrix* dA;
	matrix* dB;
	matrix* dC;	
	prepareDataForUpdateTest(&dA, &dB, &dC);
	
	float* dCoeffs = deviceMallocCoeffsAndZero();
	
	float* element;
	cudaMalloc((void**)&element, sizeof(float));

	calculateElement<<<1,1>>>(element, 0, dA, dB);
	cudaThreadSynchronize();
	
	countCoeffsLower<<<1,N>>>(element, dB, 0, dCoeffs);
	cudaThreadSynchronize();
	
	updateLowerLeft<<<1,N*N>>>(0, dA, dC, dCoeffs);
	cudaThreadSynchronize();

	matrix* hC = downloadMatrix(dC);
	
	double actual = hC->ll[0];
	double expected = -(32 * 32) / 48.0f;
	CuAssertDblEquals(tc, expected, actual, 0.0001);
	
	actual = hC->ll[13];
	expected = -(33*44)/48.0f;
	CuAssertDblEquals(tc, expected, actual, 0.0001);
	
	free(hC);
	cudaFree(element);
	cudaFree(dCoeffs);
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);	
}

void TestUpdateLowerRight(CuTest *tc) {
	matrix* dA;
	matrix* dB;
	matrix* dC;
	prepareDataForUpdateTest(&dA, &dB, &dC);
	
	float* dCoeffs = deviceMallocCoeffsAndZero();
	
	float* element;
	cudaMalloc((void**)&element, sizeof(float));
	
	calculateElement<<<1,1>>>(element, 0, dA, dB);
	cudaThreadSynchronize();
	
	countCoeffsLower<<<1,N>>>(element, dB, 0, dCoeffs);
	cudaThreadSynchronize();
	
	updateLowerRight<<<1,N*N>>>(0, dA, dC, dCoeffs);
	cudaThreadSynchronize();
	
	matrix* hC = downloadMatrix(dC);
	
	double actual = hC->lr[0];
	double expected = 48 -(16 * 32) / 48.0f;
	CuAssertDblEquals(tc, expected, actual, 0.0001);
	
	actual = hC->lr[11];
	expected = 59 - (19*40)/48.0f;
	CuAssertDblEquals(tc, expected, actual, 0.0001);
	
	free(hC);
	cudaFree(element);
	cudaFree(dCoeffs);
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);	
}

void TestUpdateUpperRight(CuTest *tc) {
	matrix* dA;
	matrix* dB;
	matrix* dC;
	prepareDataForUpdateTest(&dA, &dB, &dC);
	
	float* dCoeffs = deviceMallocCoeffsAndZero();
	
	float* element;
	cudaMalloc((void**)&element, sizeof(float));
	
	calculateElement<<<1,1>>>(element, 0, dA, dB);
	cudaThreadSynchronize();
	
	countCoeffsUpper<<<1,N>>>(element, dA, 0, dCoeffs);
	cudaThreadSynchronize();
	
	updateUpperRight<<<1,N*N>>>(0, dB, dC, dCoeffs);
	cudaThreadSynchronize();
	
	matrix* hC = downloadMatrix(dC);
	
	double actual = hC->ur[0];
	double expected = -(16 * 16) / 48.0f;
	CuAssertDblEquals(tc, expected, actual, 0.0001);
	
	actual = hC->ur[12];
	expected = -(28*16)/48.0f;
	CuAssertDblEquals(tc, expected, actual, 0.0001);
	
	free(hC);
	cudaFree(element);
	cudaFree(dCoeffs);
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
}

void TestUpdateUpperCenter(CuTest* tc) {
	matrix* dA;
	matrix* dB;
	matrix* dC;	
	prepareDataForUpdateTest(&dA, &dB, &dC);
	
	float* dCoeffs = deviceMallocCoeffsAndZero();
	
	float* element;
	cudaMalloc((void**)&element, sizeof(float));
	
	calculateElement<<<1,1>>>(element, 0, dA, dB);
	cudaThreadSynchronize();
	
	countCoeffsUpper<<<1,N>>>(element, dA, 0, dCoeffs);
	cudaThreadSynchronize();
	
	updateUpperCenter<<<1,N*N>>>(0, dA, dB, dCoeffs);
	cudaThreadSynchronize();
	
	matrix* hA = downloadMatrix(dA);
	
	double actual = hA->ur[0];
	double expected = 0.0f;	
	CuAssertDblEquals(tc, expected, actual, 0.0001);
	
	actual = hA->ur[12];
	expected = 0.0f;
	CuAssertDblEquals(tc, expected, actual, 0.0001);
	
	actual = hA->ur[1];
	expected = 17 - (16*(49+1))/48.0f;
	CuAssertDblEquals(tc, expected, actual, 0.0001);
	
	free(hA);
	cudaFree(element);
	cudaFree(dCoeffs);
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
}

void TestUpdateLowerCenter(CuTest* tc) {
	matrix* dA;
	matrix* dB;
	matrix* dC;	
	prepareDataForUpdateTest(&dA, &dB, &dC);
	
	float* dCoeffs = deviceMallocCoeffsAndZero();
	
	float* element;
	cudaMalloc((void**)&element, sizeof(float));
	
	calculateElement<<<1,1>>>(element, 0, dA, dB);
	cudaThreadSynchronize();
	
	countCoeffsLower<<<1,N>>>(element, dA, 0, dCoeffs);
	cudaThreadSynchronize();
	
	updateLowerCenter<<<1,N*N>>>(0, dA, dB, dCoeffs);
	cudaThreadSynchronize();
	
	matrix* hB = downloadMatrix(dB);
	
	double actual = hB->ll[0];
	double expected = 0.0f;	
	CuAssertDblEquals(tc, expected, actual, 0.0001);
	
	actual = hB->ll[12];
	expected = 0.0f;
	CuAssertDblEquals(tc, expected, actual, 0.0001);

	actual = hB->ll[1];
	expected = 33 - (32*(49+1))/48.0f;
	CuAssertDblEquals(tc, expected, actual, 0.0001);
		
	free(hB);
	cudaFree(element);
	cudaFree(dCoeffs);
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
}

void TestUpdateCenter(CuTest* tc) {
	matrix* dA;
	matrix* dB;
	matrix* dC;	
	prepareDataForUpdateTest(&dA, &dB, &dC);
	
	float* dCoeffs = deviceMallocCoeffsAndZero();
	
	float* element;
	cudaMalloc((void**)&element, sizeof(float));
	
	calculateElement<<<1,1>>>(element, 0, dA, dB);
	cudaThreadSynchronize();
	
	countCoeffsCenter<<<1,N>>>(element, dA, dB, 0, dCoeffs);
	cudaThreadSynchronize();
	
	updateCenter<<<1,N*(N-1)>>>(0, dA, dB, dCoeffs);
	cudaThreadSynchronize();
	
	matrix* hA = downloadMatrix(dA);
	matrix* hB = downloadMatrix(dB);
	
	double actual = hA->lr[0] + hB->ul[0];
	double expected = 48.0f;	
	CuAssertDblEquals(tc, expected, actual, 0.0001);
	
	actual = hA->lr[4] + hB->ul[4];
	expected = 0.0f;
	CuAssertDblEquals(tc, expected, actual, 0.0001);

	actual = hA->lr[12] + hB->ul[12];
	expected = 0;
	CuAssertDblEquals(tc, expected, actual, 0.0001);
		
	actual = hA->lr[5] + hB->ul[5];
	expected = 58 - (56 * 50)/48.0f;
	CuAssertDblEquals(tc, expected, actual, 0.0001);
	
	free(hB);
	free(hA);
	cudaFree(element);
	cudaFree(dCoeffs);
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
}

void TestUpdateLeftCenter(CuTest* tc) {
	matrix* dA;
	matrix* dB;
	matrix* dC;	
	prepareDataForUpdateTest(&dA, &dB, &dC);
	
	float* dCoeffs = deviceMallocCoeffsAndZero();
	
	float* element;
	cudaMalloc((void**)&element, sizeof(float));
	
	calculateElement<<<1,1>>>(element, 0, dA, dB);
	cudaThreadSynchronize();
	
	countCoeffsCenter<<<1,N>>>(element, dA, dB, 0, dCoeffs);
	cudaThreadSynchronize();
	
	updateLeftCenter<<<1,N*(N-1)>>>(0, dA, dCoeffs);
	cudaThreadSynchronize();
	
	matrix* hA = downloadMatrix(dA);

	double actual = hA->ll[0];
	double expected = 32.0f;	
	CuAssertDblEquals(tc, expected, actual, 0.0001);
	
	actual = hA->ll[12];
	expected = 44 - (72*32)/48.0f;
	CuAssertDblEquals(tc, expected, actual, 0.0001);
	
	free(hA);
	cudaFree(element);
	cudaFree(dCoeffs);
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
}

void TestUpdateRightCenter(CuTest* tc) {
	matrix* dA;
	matrix* dB;
	matrix* dC;	
	prepareDataForUpdateTest(&dA, &dB, &dC);
	
	float* dCoeffs = deviceMallocCoeffsAndZero();
	
	float* element;
	cudaMalloc((void**)&element, sizeof(float));
	
	calculateElement<<<1,1>>>(element, 0, dA, dB);
	cudaThreadSynchronize();
	
	countCoeffsCenter<<<1,N>>>(element, dA, dB, 0, dCoeffs);
	cudaThreadSynchronize();
	
	updateRightCenter<<<1,N*(N-1)>>>(0, dB, dCoeffs);
	cudaThreadSynchronize();
	
	matrix* hB = downloadMatrix(dB);

	double actual = hB->ur[0];
	double expected = 16.0f;	
	CuAssertDblEquals(tc, expected, actual, 0.0001);
	
	actual = hB->ur[12];
	expected = 28 - (72*16)/48.0f;
	CuAssertDblEquals(tc, expected, actual, 0.0001);
	
	free(hB);
	cudaFree(element);
	cudaFree(dCoeffs);
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
}

void TestUpdateBCenter(CuTest* tc) {
	matrix* dA;
	matrix* dB;
	matrix* dC;	
	prepareDataForUpdateTest(&dA, &dB, &dC);
	
	float* dCoeffs = deviceMallocCoeffsAndZero();
	
	float* element;
	cudaMalloc((void**)&element, sizeof(float));
	
	calculateElement<<<1,1>>>(element, 0, dA, dB);
	cudaThreadSynchronize();
	
	countCoeffsCenter<<<1,N>>>(element, dA, dB, 0, dCoeffs);
	cudaThreadSynchronize();
	
	updateBCenter<<<1,N-1>>>(0, dA, dB, dCoeffs);
	cudaThreadSynchronize();
	
	matrix* hA = downloadMatrix(dA);
	matrix* hB = downloadMatrix(dB);

	double actual = hA->lb[0] + hB->ub[0];
	double expected = 4.0f;	
	CuAssertDblEquals(tc, expected, actual, 0.0001);
	
	actual = hA->lb[1] + hB->ub[1];
	expected = 6 - (56*4)/48.0f;
	CuAssertDblEquals(tc, expected, actual, 0.0001);
	
	actual = hA->lb[3] + hB->ub[3];
	expected = 10 - (72*4)/48.0f;
	CuAssertDblEquals(tc, expected, actual, 0.0001);
	
	free(hA);
	free(hB);
	cudaFree(element);
	cudaFree(dCoeffs);
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
}

void TestUpdateBLower(CuTest* tc) {
	matrix* dA;
	matrix* dB;
	matrix* dC;	
	prepareDataForUpdateTest(&dA, &dB, &dC);
	
	float* dCoeffs = deviceMallocCoeffsAndZero();
	
	float* element;
	cudaMalloc((void**)&element, sizeof(float));
	
	calculateElement<<<1,1>>>(element, 0, dA, dB);
	cudaThreadSynchronize();
	
	countCoeffsLower<<<1,N>>>(element, dA, 0, dCoeffs);
	cudaThreadSynchronize();
	
	updateBLower<<<1,N>>>(0, dA, dB, dC, dCoeffs);
	cudaThreadSynchronize();
	
	matrix* hC = downloadMatrix(dC);

	double actual = hC->lb[0];
	double expected = 4 - (32*4)/48.0f;	
	CuAssertDblEquals(tc, expected, actual, 0.0001);
	
	actual = hC->lb[3];
	expected = 7 - (44*4)/48.0f;
	CuAssertDblEquals(tc, expected, actual, 0.0001);
	
	free(hC);
	cudaFree(element);
	cudaFree(dCoeffs);
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
}

void TestUpdateBUpper(CuTest* tc) {
	matrix* dA;
	matrix* dB;
	matrix* dC;	
	prepareDataForUpdateTest(&dA, &dB, &dC);
	
	float* dCoeffs = deviceMallocCoeffsAndZero();
	
	float* element;
	cudaMalloc((void**)&element, sizeof(float));
	
	calculateElement<<<1,1>>>(element, 0, dA, dB);
	cudaThreadSynchronize();
	
	countCoeffsUpper<<<1,N>>>(element, dA, 0, dCoeffs);
	cudaThreadSynchronize();
	
	updateBUpper<<<1,N>>>(0, dA, dB, dC, dCoeffs);
	cudaThreadSynchronize();
	
	matrix* hC = downloadMatrix(dC);

	double actual = hC->ub[0];
	double expected = - (16*4)/48.0f;	
	CuAssertDblEquals(tc, expected, actual, 0.0001);
	
	actual = hC->ub[3];
	expected = 3 - (28*4)/48.0f;
	CuAssertDblEquals(tc, expected, actual, 0.0001);
	
	free(hC);
	cudaFree(element);
	cudaFree(dCoeffs);
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
}


CuSuite* SolverGetSuite() {
	CuSuite* suite = CuSuiteNew();
	SUITE_ADD_TEST(suite, TestPrepareDataForUpdateTest);
	SUITE_ADD_TEST(suite, TestCalculateElement);
	
	SUITE_ADD_TEST(suite, TestCountUpper);
	SUITE_ADD_TEST(suite, TestCountLower);
	SUITE_ADD_TEST(suite, TestCountCenter);
	
	SUITE_ADD_TEST(suite, TestUpdateUpperLeft);
	SUITE_ADD_TEST(suite, TestUpdateLowerRight);
	SUITE_ADD_TEST(suite, TestUpdateLowerLeft);
	SUITE_ADD_TEST(suite, TestUpdateUpperRight); 
	
	SUITE_ADD_TEST(suite, TestUpdateUpperCenter);
	SUITE_ADD_TEST(suite, TestUpdateLowerCenter);
	
	SUITE_ADD_TEST(suite, TestUpdateCenter);
	SUITE_ADD_TEST(suite, TestUpdateLeftCenter);
	SUITE_ADD_TEST(suite, TestUpdateRightCenter);
	
	SUITE_ADD_TEST(suite, TestUpdateBCenter);
	SUITE_ADD_TEST(suite, TestUpdateBLower);
	SUITE_ADD_TEST(suite, TestUpdateBUpper);
	
	return suite;
}
