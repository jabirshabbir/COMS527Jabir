#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "util/img.h"
#include "neural/activations.h"
#include "neural/nn.h"
#include "matrix/matrix.h"
#include "matrix/ops.h"
#include <time.h>
//using namespace std;

int main() {
	srand(time(NULL));
	clock_t starttime = clock() / (CLOCKS_PER_SEC / 1000);
	//TRAINING
	 time_t start,end;
	 int number_imgs = 10000;
	 Img** imgs = csv_to_imgs("./MNIST_data_train.csv", number_imgs);
	 NeuralNetwork* net = network_create(784, 300, 10, 0.1);
	 start = time(&start);
	 network_train_batch_imgs(net, imgs, number_imgs);
	 end = time(&end);
	 clock_t endtime = clock() / (CLOCKS_PER_SEC / 1000);
	 network_save(net, "testing_net");
	 printf("%f", (double)(end - start));

	// PREDICTING
	 int number_imgs = 3000;
	 Img** imgs = csv_to_imgs("data/mnist_test.csv", number_imgs);
	 NeuralNetwork* net = network_load("testing_net");
	 double score = network_predict_imgs(net, imgs, 1000);
	 printf("Score: %1.5f\n", score);

	// imgs_free(imgs, number_imgs);
	// network_free(net);
	return 0;
}
