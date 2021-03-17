#include "pca.h"

int main() {

	int *train_Mat, *test_Mat;
	int n_train = 163, n_test = 50; //number of train and test images
	int img_size = 64; // size of image
	int k = 32;       // number of principal components
	int dimension = img_size * img_size;
	int train_MatBuffer = n_train * dimension * sizeof(int);
	int test_MatBuffer = n_test * dimension * sizeof(int);

	//Read training set
	printf("Loading train data...\n");
	train_Mat = (int*)malloc(train_MatBuffer);
	ReadFile("train_matrix.txt", train_Mat, n_train, dimension);

	//Read test set
	printf("Loading test data...\n");
	test_Mat = (int*)malloc(test_MatBuffer);
	ReadFile("test_matrix.txt", test_Mat, n_test, dimension);
	printf("N_train:%d, N_test:%d\n\n", n_train, n_test);

	int *d_train_Mat, *d_test_Mat;
	float *d_mean;
	int imgBuffer = dimension * sizeof(float);
	cudaMalloc((void**)&d_train_Mat, train_MatBuffer);
	cudaMalloc((void**)&d_test_Mat, test_MatBuffer);
	cudaMalloc((void**)&d_mean, imgBuffer);

	//Copy the training and test matrices from Host to Device
	printf("Copying training and test matrices from Host to Device...\n\n");
	cudaMemcpy(d_train_Mat, train_Mat, train_MatBuffer, cudaMemcpyHostToDevice);
	cudaMemcpy(d_test_Mat, test_Mat, test_MatBuffer, cudaMemcpyHostToDevice);

	printf("Calculating principal components for training data...\n");
	printf("N_PC:%d\n\n", k);
	//Calculate the mean vector of the training matrix
	cuda_Mean <<<32, 128 >>>(d_train_Mat, n_train, dimension, d_mean);

	//Compute the difference between each image vector and the mean vector
	dim3 block1(128, 1);
	dim3 grid1(dimension / 128, n_train);
	float *d_X;
	int Xbuffer = n_train * dimension * sizeof(float);
	cudaMalloc((void**)&d_X, Xbuffer);
	cuda_SubtractMean << <grid1, block1 >> >(d_train_Mat, d_mean, n_train, dimension, d_X);

	//Determine the covariance matrix
	int covMatBuffer = dimension * dimension * sizeof(float);
	float *d_Xt, *d_covMat;
	cudaMalloc((void**)&d_Xt, Xbuffer);
	cudaMalloc((void**)&d_covMat, covMatBuffer);
	cuda_MatTranspose << <grid1, block1 >> >(d_X, d_Xt, n_train, dimension);
	dim3 block2(32, 16);
	dim3 grid2(dimension / 32, dimension / 16);
	cuda_MatMul << <grid2, block2 >> >(d_Xt, d_X, d_covMat, dimension, n_train, dimension);

	//Find eigenvalues and eigenvectors of the covariance matrix
	float prev_lambda = 0;
	const float eps = 0.000001;
	float *Q, *normZ, *k_eig_vec, *d_Q, *d_Z, *d_W, *d_normZ;
	int normBuffer = sizeof(float);
	int eigVecBuffer = k * imgBuffer;

	Q = (float*)malloc(imgBuffer);
	normZ = (float*)malloc(normBuffer);
	k_eig_vec = (float*)malloc(eigVecBuffer);
	cudaMalloc((void**)&d_Q, imgBuffer);
	cudaMalloc((void**)&d_Z, imgBuffer);
	cudaMalloc((void**)&d_W, covMatBuffer);
	cudaMalloc((void**)&d_normZ, normBuffer);

	int block3 = 32;
	int grid3 = dimension / block3;
	int sharedMemSize = block3 * sizeof(float);
	int DestIndex;

	for (int i = 0; i < k; i++) {
		Q[0] = 1;
		for (int m = 1; m < dimension; m++) Q[m] = 0;
		cudaMemcpy(d_Q, Q, imgBuffer, cudaMemcpyHostToDevice);
		cuda_MatMul << <16, 256 >>>(d_covMat, d_Q, d_Z, dimension, dimension, 1);

		//Power Method iteration
		for (int j = 0; j < 1000; j++) {
			normZ[0] = 0;
			cudaMemcpy(d_normZ, normZ, normBuffer, cudaMemcpyHostToDevice);
			cuda_Norm2 << <grid3, block3, sharedMemSize >> >(d_Z, d_normZ, dimension);
			cudaDeviceSynchronize();

			cudaMemcpy(normZ, d_normZ, normBuffer, cudaMemcpyDeviceToHost);
			normZ[0] = sqrt(normZ[0]);
			cudaMemcpy(d_normZ, normZ, normBuffer, cudaMemcpyHostToDevice);
			cuda_Normalize << <grid3, block3, sharedMemSize >> >(d_Z, d_normZ, d_Q, dimension);
			cudaDeviceSynchronize();

			cuda_MatMul << <16, 256 >> >(d_Q, d_covMat, d_Z, 1, dimension, dimension);
			cudaDeviceSynchronize();

			normZ[0] = 0;
			cudaMemcpy(d_normZ, normZ, normBuffer, cudaMemcpyHostToDevice);
			cuda_EigenValue << <grid3, block3, sharedMemSize >> >(d_Q, d_Z, d_normZ, dimension);
			cudaDeviceSynchronize();

			cudaMemcpy(normZ, d_normZ, normBuffer, cudaMemcpyDeviceToHost);
			if (abs(prev_lambda - normZ[0]) < eps) {
				cudaMemcpy(Q, d_Q, imgBuffer, cudaMemcpyDeviceToHost);
				break;
			}
			prev_lambda = normZ[0];
		}

		//The new subspace created by k eigenvectors
		for (int p = 0; p < dimension; p++) {
			DestIndex = p * k + i;
			k_eig_vec[DestIndex] = Q[p];
		}

		//Calculate the new covariance matrix for the next eigenvalue and eigenvector
		cuda_MatMul << <grid2, block2 >> >(d_Q, d_Q, d_W, dimension, 1, dimension);
		cuda_MatMulScalar << <grid2, block2 >> >(d_W, d_W, dimension, dimension, normZ[0]);
		cuda_MatSub << <grid2, block2 >> >(d_covMat, d_W, d_covMat, dimension, dimension);
	}

	//Project the training images in the new subspace
	printf("Projecting training images onto the new subspace...\n");
	float *d_kEigVec, *d_proj_train;
	int projTrainBuffer = n_train * k * sizeof(float);
	cudaMalloc((void**)&d_kEigVec, eigVecBuffer);
	cudaMalloc((void**)&d_proj_train, projTrainBuffer);
	cudaMemcpy(d_kEigVec, k_eig_vec, eigVecBuffer, cudaMemcpyHostToDevice);
	dim3 grid4((k + block2.x - 1) / block2.x, (n_train + block2.y - 1) / block2.y);
	cuda_MatMul << <grid4, block2 >> >(d_X, d_kEigVec, d_proj_train, n_train, dimension, k);

	//Project the test images in the new subspace
	printf("Projecting test images onto the new subspace...\n\n");
	float *d_X1;
	cudaMalloc((void**)&d_X1, test_MatBuffer);
	dim3 grid5(dimension / 128, n_test);
	cuda_SubtractMean << <grid5, block1 >> >(d_test_Mat, d_mean, n_test, dimension, d_X1);

	float *d_proj_test;
	int projTestBuffer = n_test * k * sizeof(float);
	cudaMalloc((void**)&d_proj_test, projTestBuffer);
	dim3 grid6((k + block2.x - 1) / block2.x, (n_test + block2.y - 1) / block2.y);
	cuda_MatMul << <grid6, block2 >> >(d_X1, d_kEigVec, d_proj_test, n_test, dimension, k);

	//Copy projected training and test matrices from Device to Host
	printf("Copying projected training and test images from Devive to Host...\n\n");
	float *project_train_img, *project_test_img;
	project_train_img = (float*)malloc(projTrainBuffer);
	project_test_img = (float*)malloc(projTestBuffer);
	cudaMemcpy(project_train_img, d_proj_train, projTrainBuffer, cudaMemcpyDeviceToHost);
	cudaMemcpy(project_test_img, d_proj_test, projTestBuffer, cudaMemcpyDeviceToHost);

	//Test image predictions
	printf("Generating test image predictions...\n");
	int *predictions;
	int prediction_Buffer = n_test * sizeof(int);
	predictions = (int*)malloc(prediction_Buffer);
	Predict(project_train_img, project_test_img, n_train, n_test, k, predictions);

	WriteFile("test_predictions.txt", n_test, predictions);
	printf("Predictions are saved in build/Src/test_predictions.txt...\n");

	return 0;

}
