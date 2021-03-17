#include <iostream>
#include <stdio.h>
#include <tchar.h>
#include <stdio.h>
#include <math.h>
#include <iostream>

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

//CPU FUNCTIONS
//--------------------------------------------------------------------------------
void Matrix_Addition(float* A, float* B, int Nrows, int Ncols, float* C) {
	int col = 0;
	int row = 0;
	int DestIndex = 0;
	for (col = 0; col < Ncols; col++) {
		for (row = 0; row < Nrows; row++) {
			DestIndex = row * Ncols + col;
			C[DestIndex] = A[DestIndex] + B[DestIndex];
		}
	}
}

void Matrix_Subtraction(float* A, float* B, int Nrows, int Ncols, float* C) {
	int col = 0;
	int row = 0;
	int DestIndex = 0;
	for (col = 0; col < Ncols; col++) {
		for (row = 0; row < Nrows; row++) {
			DestIndex = row * Ncols + col;
			C[DestIndex] = A[DestIndex] - B[DestIndex];
		}
	}
}

void Matrix_Multiplication(float *A, float *B, int m, int n, int p, float *C) {
	float tmp;
	int outidx;
	for (int col = 0; col < p; col++) {
		for (int row = 0; row < m; row++) {
			outidx = row * p + col;
			tmp = 0;
			for (int idx = 0; idx < n; idx++) {
				tmp += A[row*n + idx] * B[idx*p + col];
			}
			C[outidx] = tmp;
		}
	}
}

void Matrix_MultiplicationScalar(float *A, int Nrows, int Ncols, float lambda, float *B) {
	int DestIndex;
	for (int col = 0; col < Ncols; col++) {
		for (int row = 0; row < Nrows; row++) {
			DestIndex = row * Ncols + col;
			B[DestIndex] = lambda * A[DestIndex];
		}
	}
}

void Matrix_Transpose(float *A, int Nrows, int Ncols, float *B) {
	int col = 0;
	int row = 0;
	for (col = 0; col < Ncols; col++) {
		for (row = 0; row < Nrows; row++) {
			B[col * Nrows + row] = A[row * Ncols + col];
		}
	}
}

float Matrix_Norm(float *A, int Nrows, int Ncols) {
	float tmp = 0;
	float Norm;
	int DestIndex;
	for (int col = 0; col < Ncols; col++) {
		for (int row = 0; row < Nrows; row++) {
			DestIndex = row * Ncols + col;
			tmp = tmp + A[DestIndex] * A[DestIndex];
		}
	}
	Norm = sqrt(tmp);
	return Norm;
}

//Principal Component Analysis
void PowerMethod(float *A, int n, float eps, float *eigVec, float *lambda) {
	float *Q;
	float *prevQ;
	float *Z;
	float *StepVec;
	float *QtA;
	float norm2z = 0;
	float dist = 1;
	int Buffer = n * sizeof(float);

	Q = (float*)malloc(Buffer);
	prevQ = (float*)malloc(Buffer);
	Z = (float*)malloc(Buffer);
	StepVec = (float*)malloc(Buffer);
	QtA = (float*)malloc(Buffer);

	Q[0] = 1;
	for (int i = 1; i < n; i++) {
		Q[i] = 0;
	}

	do {
		for (int i = 0; i < n; i++) {
			prevQ[i] = Q[i];
		}
		Matrix_Multiplication(A, Q, n, n, 1, Z);
		norm2z = Matrix_Norm(Z, n, 1);
		for (int i = 0; i < n; i++) {
			Q[i] = Z[i] / norm2z;
		}
		Matrix_Subtraction(Q, prevQ, n, 1, StepVec);
		dist = Matrix_Norm(StepVec, n, 1);
	} while (dist > eps);

	for (int i = 0; i < n; i++) {
		eigVec[i] = Q[i];
	}

	Matrix_Multiplication(Q, A, 1, n, n, QtA);
	Matrix_Multiplication(QtA, Q, 1, n, 1, lambda);

	free(Q);
	free(prevQ);
	free(Z);
	free(StepVec);
	free(QtA);
}

void trainPCA(int *trainMat, int size, int Ntrain, int k, float *project_train_img, float *k_eig_vec, float *mean) {
	int dimension = size;
	float tmp;
	int DestIndex;
	int trainMatBuffer = Ntrain * dimension * sizeof(int);

	for (int i = 0; i < dimension; i++) {
		tmp = 0;
		for (int j = 0; j < Ntrain; j++) {
			tmp = tmp + trainMat[j*dimension + i];
		}
		mean[i] = tmp / Ntrain;
	}

	// Subtract mean vector
	float *X;
	int Xbuffer = Ntrain * dimension * sizeof(float);
	X = (float*)malloc(Xbuffer);
	for (int i = 0; i < Ntrain; i++) {
		for (int j = 0; j < dimension; j++) {
			DestIndex = i * dimension + j;
			X[DestIndex] = trainMat[DestIndex] - mean[j];
		}
	}

	// Compute covariance matrix
	float *covMat;
	float *X1;
	int covMatBuffer = dimension * dimension * sizeof(float);
	covMat = (float*)malloc(covMatBuffer);
	X1 = (float*)malloc(Xbuffer);
	Matrix_Transpose(X, Ntrain, dimension, X1);
	Matrix_Multiplication(X1, X, dimension, Ntrain, dimension, covMat);

	int imgBuffer;
	imgBuffer = dimension * sizeof(float);

	float *eigVal;
	float *eigVec;
	float *B;
	float *V;
	float *VVt;
	float *lambda;

	B = (float*)malloc(covMatBuffer);
	V = (float*)malloc(imgBuffer);
	VVt = (float*)malloc(covMatBuffer);
	eigVal = (float*)malloc(k * sizeof(float));
	eigVec = (float*)malloc(imgBuffer);
	lambda = (float*)malloc(sizeof(float));
	lambda[0] = 1;

	for (int i = 0; i < dimension; i++) {
		for (int j = 0; j < dimension; j++) {
			DestIndex = i * dimension + j;
			B[DestIndex] = covMat[DestIndex];
		}
	}
	for (int i = 0; i < k; i++) {
		PowerMethod(B, dimension, 0.00001, eigVec, lambda);

		for (int p = 0; p < dimension; p++) {
			DestIndex = p * k + i;
			k_eig_vec[DestIndex] = eigVec[p];
		}

		eigVal[i] = lambda[0];

		Matrix_Multiplication(eigVec, eigVec, dimension, 1, dimension, VVt);
		Matrix_MultiplicationScalar(VVt, dimension, dimension, eigVal[i], VVt);
		Matrix_Subtraction(B, VVt, dimension, dimension, B);
	}

	Matrix_Multiplication(X, k_eig_vec, Ntrain, dimension, k, project_train_img);

	free(X);
	free(covMat);
	free(X1);
	free(eigVal);
	free(B);
	free(V);
	free(VVt);
}

//Project test images onto the new subspace
void testPCA(int *testMat, float *k_eig_vec, float *mean, int Ntest, int imgSize, int k, float *project_test_img) {
	int DestIndex;
	int dimension = imgSize;
	float *X;
	int Xbuffer = Ntest * dimension * sizeof(float);
	X = (float*)malloc(Xbuffer);
	for (int i = 0; i < Ntest; i++) {
		for (int j = 0; j < dimension; j++) {
			DestIndex = i * dimension + j;
			X[DestIndex] = testMat[DestIndex] - mean[j];
		}
	}

	Matrix_Multiplication(X, k_eig_vec, Ntest, dimension, k, project_test_img);

	free(X);
}

//Generate test image predictions
void Predict(float *project_train_img, float *project_test_img, int Ntrain, int Ntest, int dimension, int *recognized_img) {
	int DestIndex;
	float *test, *train, *D;
	float distance = 0, min;

	int buffer = dimension * sizeof(float);
	test = (float*)malloc(buffer);
	train = (float*)malloc(buffer);
	D = (float*)malloc(buffer);

	for (int i = 0; i < Ntest; i++) {
		min = 100000;
		for (int j = 0; j < dimension; j++) {
			DestIndex = i * dimension + j;
			test[j] = project_test_img[DestIndex];
		}

		for (int m = 0; m < Ntrain; m++) {
			for (int n = 0; n < dimension; n++) {
				DestIndex = m * dimension + n;
				train[n] = project_train_img[DestIndex];
			}
			Matrix_Subtraction(train, test, 1, dimension, D);
			distance = Matrix_Norm(D, 1, dimension);
			if (distance < min) {
				min = distance;
				recognized_img[i] = m + 1;
			}
		}
	}

	free(test);
	free(train);
	free(D);
}
//---------------------------------------------------------------------