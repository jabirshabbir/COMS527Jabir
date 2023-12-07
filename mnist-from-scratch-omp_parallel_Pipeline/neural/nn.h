#pragma once

#include "../matrix/matrix.h"
#include "../util/img.h"

extern Matrix*** all_activations;
extern int z;
extern Matrix* activate;

typedef struct {
	int input;
	int hidden1;
	int hidden2;
	int hidden3;
	int output;
	double learning_rate;
	Matrix* hidden_weights_1;
	Matrix* hidden_weights_2;
	Matrix* hidden_weights_3;
	Matrix* output_weights;
} NeuralNetwork;

NeuralNetwork* network_create(int input, int hidden1,int hidden2,int hidden3, int num_hidden_layers, int output, double lr);
void network_train(NeuralNetwork* net, Matrix* input_data, Matrix* output_data,int pipeline_stage,int input_id);
void network_train_batch_imgs(NeuralNetwork* net, Img** imgs, int batch_size);
Matrix* network_predict_img(NeuralNetwork* net, Img* img);
double network_predict_imgs(NeuralNetwork* net, Img** imgs, int n);
Matrix* network_predict(NeuralNetwork* net, Matrix* input_data);
void network_save(NeuralNetwork* net, char* file_string);
NeuralNetwork* network_load(char* file_string);
void network_print(NeuralNetwork* net);
void network_free(NeuralNetwork* net);