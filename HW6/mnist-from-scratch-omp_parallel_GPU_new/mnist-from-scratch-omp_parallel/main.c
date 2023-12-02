#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "util/img.h"
#include "neural/activations.h"
#include "neural/nn.h"
#include "matrix/matrix.h"
#include "matrix/ops.h"

int main() {
	srand(time(NULL));

	//TRAINING
	 int number_imgs = 10000;
	 Img** imgs = csv_to_imgs("C:/Users/jabir/Downloads/mnist-from-scratch-omp_parallel/mnist-from-scratch-omp_parallel/MNIST_data_train.csv", number_imgs);
	 NeuralNetwork* net = network_create(784, 300, 10, 0.1);
	 clock_t starttime = clock() / (CLOCKS_PER_SEC / 1000);
	 network_train_batch_imgs(net, imgs, number_imgs);
	 clock_t endtime = clock() / (CLOCKS_PER_SEC / 1000);
	 network_save(net, "testing_net");

	// PREDICTING
	/*int number_imgs = 3000;
	Img** imgs = csv_to_imgs("data/mnist_test.csv", number_imgs);
	NeuralNetwork* net = network_load("testing_net");
	double score = network_predict_imgs(net, imgs, 1000);
	printf("Score: %1.5f\n", score);
	*/
	imgs_free(imgs, number_imgs);
	network_free(net);
	printf("%d", (endtime - starttime));

	return 0;
}