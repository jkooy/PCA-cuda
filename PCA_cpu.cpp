#include <time.h>
#include "PCA_cpu.h"

int main() {

	clock_t tStart = clock();

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
	printf("Loading test data...\n\n");
	test_Mat = (int*)malloc(test_MatBuffer);
	ReadFile("test_matrix.txt", test_Mat, n_test, dimension);
	printf("N_train:%d, N_test:%d\n\n", n_train, n_test);

	// Traing PCA
	printf("Finding principal components based on train images...\n");
	float *project_train_img, *k_eig_vec, *mean;
	int projTrainBuffer = n_train * k * sizeof(float);
	int eigVecBuffer = dimension * k * sizeof(float);
	int meanBuffer = dimension * sizeof(float);
	project_train_img = (float*)malloc(projTrainBuffer);
	k_eig_vec = (float*)malloc(eigVecBuffer);
	mean = (float*)malloc(meanBuffer);
	trainPCA(train_Mat, dimension, n_train, k, project_train_img, k_eig_vec, mean);

	//Project test images onto the new subspace
	printf("Projecting test images onto the new subspace...\n\n");
	float *project_test_img;
	int projTestBuffer = n_test * k * sizeof(float);
	project_test_img = (float*)malloc(projTestBuffer);
	testPCA(test_Mat, k_eig_vec, mean, n_test, dimension, k, project_test_img);

	//Test image predictions
	printf("Generating test image predictions...\n");
	int *predictions;
	int prediction_Buffer = n_test * sizeof(int);
	predictions = (int*)malloc(prediction_Buffer);
	Predict(project_train_img, project_test_img, n_train, n_test, k, predictions);

	WriteFile("test_predictions.txt", n_test, predictions);
	printf("Predictions are saved in Src/test_predictions.txt...\n");


	printf("Time taken: %.2fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

	return 0;
}

