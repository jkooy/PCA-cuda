#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <string>
#include <filesystem>
#include<direct.h>
#include <stdlib.h>

// read train and text images from txt file
int ReadFile(char* FileName, int* Mat, int N, int dimension) {
	FILE *fp;
	int DestIndex;
	fp = fopen(FileName, "r");
	if (fp == NULL) {
		printf("Error in opening file\n");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < dimension; j++) {
			DestIndex = i * dimension + j;
			fscanf(fp, "%d", &Mat[DestIndex]);
		}
	}
	fclose(fp);
	return 0;
}

// write predictions into txt file
int WriteFile(char* FileName, int N, int* predictions) {
	FILE *fp;
	fp = fopen(FileName, "wb");
	if (fp == NULL) {
		printf("Error in opening file\n");
		exit(EXIT_FAILURE);
	}
	fprintf(fp, "Test Image\t\tPrediction\n");
	for (int i = 0; i < N; i++) {
		fprintf(fp, "%d \t\t\t\t %d\n", i + 1, predictions[i]);
	}
	fclose(fp);
	return 0;
}

//DEVICE FUNCTIONS
//------------------------------------------------------------------------------------
// device function for matrix subtraction
__global__ void cuda_MatSub(float* A, float* B, float* C, int M, int N) {
	int ind, idx, idy;
	idx = threadIdx.x + blockIdx.x * blockDim.x;
	idy = threadIdx.y + blockIdx.y * blockDim.y;
	if ((idx < N) && (idy < M)) {
		ind = N * idy + idx;
		C[ind] = A[ind] - B[ind];
	}
}

//device function for matrix multiplication
__global__ void cuda_MatMul(float *A, float *B, float *C, int m, int n, int p) { 
  int row = blockIdx.y * blockDim.y + threadIdx.y; 
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float S = 0;
  if (col < p && row < m) {
    for(int i = 0; i < n; i++) {
      S += A[row * n + i] * B[i * p + col];
    }
    C[row * p + col] = S;
  }
}

//device function for matrix mean
__global__ void cuda_Mean(int *train_Mat, int N, int dimension, float *mean) {
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	float tmp = 0;
	int DestIndex;
	for (int i = 0; i < N; i++) {
		DestIndex = i * dimension + ind;
		tmp = tmp + train_Mat[DestIndex];
	}
	mean[ind] = tmp / N;
}

//device function to subtract mean from matrix
__global__ void cuda_SubtractMean(int *Mat, float *mean, int N, int dimension, float *X) {
	int ind, idx, idy;
	idx = threadIdx.x + blockIdx.x * blockDim.x;
	idy = threadIdx.y + blockIdx.y * blockDim.y;
	if ((idx < dimension) && (idy < N)) {
		ind = dimension * idy + idx;
		X[ind] = Mat[ind] - mean[idx];
	}
}

//device function for matrix transpose
__global__ void cuda_MatTranspose(float *A, float *B, int M, int N)
{
	unsigned int idx, idy, pos, trans_pos;
	idx = blockIdx.x * blockDim.x + threadIdx.x;
	idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx < N && idy < M) {
		pos = idy * N + idx;
		trans_pos = idx * M + idy;
		B[trans_pos] = A[pos];
	}
}

//device function to multiply a matrix with a scalar
__global__ void cuda_MatMulScalar(float *A, float *B, int M, int N, float lambda) {
	int ind, idx, idy;
	idx = threadIdx.x + blockIdx.x * blockDim.x;
	idy = threadIdx.y + blockIdx.y * blockDim.y;
	if ((idx < N) && (idy < M)) {
		ind = N * idy + idx;
		B[ind] = A[ind] * lambda;
	}
}

//device funtion for matrix norm computation
__global__ void cuda_Norm2(float *A, float *norm, int N) {
	extern __shared__ float sdata[];
	unsigned int ind = threadIdx.x;
	unsigned int gid = ind + blockIdx.x * blockDim.x;

	if (gid < N) sdata[ind] = A[gid];
	else sdata[ind] = 0;
	__syncthreads();

	sdata[ind] = sdata[ind] * sdata[ind];
	__syncthreads();

	for (unsigned int stride = blockDim.x / 2; stride > 0; stride = stride >> 1) {
		if (ind < stride) sdata[ind] = sdata[ind] + sdata[ind + stride];
		__syncthreads();
	}

	if (ind == 0) atomicAdd(norm, sdata[0]);
}

//gpu function to normalize matrix
__global__ void cuda_Normalize(float *A, float *norm, float *B, int N) {
	extern __shared__ float sdata[];
	unsigned int ind = threadIdx.x;
	unsigned int gid = ind + blockIdx.x * blockDim.x;

	if (ind == 0) sdata[0] = norm[0];
	__syncthreads();

	if (gid < N) B[gid] = A[gid] / sdata[0];
}

//gpu function to compute eigen values
__global__ void cuda_EigenValue(float *A, float *B, float *eigVal, int N) {
	extern __shared__ float sdata[];
	unsigned int ind = threadIdx.x;
	unsigned int gid = ind + blockIdx.x * blockDim.x;

	if (gid < N) sdata[ind] = A[gid] * B[gid];
	else sdata[ind] = 0;
	__syncthreads();

	for (unsigned int stride = blockDim.x / 2; stride > 0; stride = stride >> 1) {
		if (ind < stride) sdata[ind] = sdata[ind] + sdata[ind + stride];
		__syncthreads();
	}
	if (ind == 0) atomicAdd(eigVal, sdata[0]);
}

//------------------------------------------------------------------------------------

//CPU FUNCTIONS
//------------------------------------------------------------------------------------

float Norm2(float *A, int M, int N) {
	float tmp = 0, Norm;
	int DestIndex;
	for (int col = 0; col < N; col++) {
		for (int row = 0; row < M; row++) {
			DestIndex = row * N + col;
			tmp = tmp + A[DestIndex] * A[DestIndex];
		}
	}
	Norm = sqrt(tmp);
	return Norm;
}

//matrix subtraction
void MatSub(float* A, float* B, int M, int N, float* C) {
	int col = 0, row = 0, DestIndex = 0;
	for (col = 0; col < N; col++) {
		for (row = 0; row < M; row++) {
			DestIndex = row * N + col;
			C[DestIndex] = A[DestIndex] - B[DestIndex];
		}
	}
}

//test image predictions
void Predict(float *project_train_img, float *project_test_img, int n_train, int n_test, int dimension, int *predictions) {
	int DestIndex;
	float *test, *train, *D;
	float distance = 0, min;

	int buffer = dimension * sizeof(float);
	test = (float*)malloc(buffer);
	train = (float*)malloc(buffer);
	D = (float*)malloc(buffer);

	for (int i = 0; i < n_test; i++) {
		min = 100000;
		for (int j = 0; j < dimension; j++) {
			DestIndex = i * dimension + j;
			test[j] = project_test_img[DestIndex];
		}

		for (int m = 0; m < n_train; m++) {
			for (int n = 0; n < dimension; n++) {
				DestIndex = m * dimension + n;
				train[n] = project_train_img[DestIndex];
			}
			MatSub(train, test, 1, dimension, D);
			distance = Norm2(D, 1, dimension);
			if (distance < min) {
				min = distance;
				predictions[i] = m + 1;
			}
		}
	}

	free(test);
	free(train);
	free(D);
}