#include "nn.h"
#include <sys/stat.h>
//#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "../matrix/ops.h"
#include "../neural/activations.h"

#define MAXCHAR 1000

// 784, 300, 10
NeuralNetwork* network_create(int input, int hidden, int output, double lr) {
	
	NeuralNetwork* net = malloc(sizeof(NeuralNetwork));
	//Matrix** hidden_weights = malloc(sizeof(Matrix*) * hidden);
	//int* hidden_layers = malloc(sizeof(int) * (nu + 1));
	int* hidden_layers = malloc(sizeof(int) * (hidden + 1));
	hidden_layers[0] = input;
	hidden_layers[1] = 300;
	hidden_layers[2] = 200;
	hidden_layers[3] = 100;
	hidden_layers[4] = 50;
	hidden_layers[5] = 60;
	hidden_layers[6] = 70;
	hidden_layers[7] = 40;
	hidden_layers[8] = 20;
	hidden_layers[9] = 90;
	hidden_layers[10] = 130;
	net->input = input;
	//net->hidden = hidden;
	net->output = output;
	net->learning_rate = lr;
	net->hidden_weights = malloc(sizeof(Matrix*)*hidden);
	for (int i = 0;i < hidden;i++)
	{
		Matrix* hidden_layer = matrix_create(hidden_layers[i+1], hidden_layers[i]);
		matrix_randomize(hidden_layer, hidden_layers[i+1]);
		net->hidden_weights[i] = hidden_layer;
	}
	//Matrix* hidden_layer = matrix_create(hidden, input);
	Matrix* output_layer = matrix_create(output, hidden_layers[hidden]);
	//matrix_randomize(hidden_layer, hidden);
	matrix_randomize(output_layer, output);
	//net->hidden_weights = hidden_layer;
	net->output_weights = output_layer;
	return net;
}

void network_train(NeuralNetwork* net, Matrix* input, Matrix* output,int hidden)
{
	// Feed forward

	Matrix** hidden_outputs_stored = malloc(sizeof(Matrix*)*(hidden+1));

	Matrix** hidden_errors_stored = malloc(sizeof(Matrix*) * (hidden));
	Matrix* next_layer_input = input;
	for (int i = 0;i < hidden;i++)
	{
		Matrix* hidden_inputs = dot(net->hidden_weights[i], next_layer_input);
		next_layer_input = apply(sigmoid, hidden_inputs);
		//Matrix* final_inputs = dot(net->output_weights, hidden_outputs);
		//next_layer_input = apply(sigmoid, next_layer_input);
		hidden_outputs_stored[i] = next_layer_input;
	}

	Matrix* final_outputs = dot(net->output_weights, next_layer_input);
	final_outputs = apply(sigmoid, final_outputs);
	//Matrix* final_inputs = dot(net->output_weights, hidden_outputs);
	//Matrix* next_layer_input = apply(sigmoid, next_layer_input);


	//backpropagate
	// Find errors

	Matrix* output_errors = subtract(output, final_outputs);
	Matrix* transposed_mat = transpose(net->output_weights);
	Matrix* hidden_errors = dot(transposed_mat, output_errors);
	matrix_free(transposed_mat);


	Matrix* sigmoid_primed_mat = sigmoidPrime(final_outputs);
	Matrix* multiplied_mat = multiply(output_errors, sigmoid_primed_mat);
	
	Matrix* t = hidden_outputs_stored[hidden - 1];
	transposed_mat = transpose(t);
	Matrix* dot_mat = dot(multiplied_mat, transposed_mat);
	Matrix* scaled_mat = scale(net->learning_rate, dot_mat);
	Matrix* added_mat = add(net->output_weights, scaled_mat);

	matrix_free(net->output_weights); // Free the old weights before replacing
	net->output_weights = added_mat;

	matrix_free(sigmoid_primed_mat);
	matrix_free(multiplied_mat);
	matrix_free(transposed_mat);
	matrix_free(dot_mat);
	matrix_free(scaled_mat);

	// Reusing variables after freeing memory

	for (int i = hidden - 1;i >= 0;i--)
	{
		Matrix* hidden_outputs = hidden_outputs_stored[i];
		sigmoid_primed_mat = sigmoidPrime(hidden_outputs);
		multiplied_mat = multiply(hidden_errors, sigmoid_primed_mat);
		Matrix* prev_hidden_output = NULL;
		if (i != 0)
		{
			prev_hidden_output = hidden_outputs_stored[i - 1];
		}
		else
		{
			prev_hidden_output = input;
		}
		
		hidden_errors = dot(transpose(net->hidden_weights[i]), hidden_errors);
		
		hidden_errors_stored[i] = hidden_errors;

		transposed_mat = transpose(prev_hidden_output);
		dot_mat = dot(multiplied_mat, transposed_mat);
		scaled_mat = scale(net->learning_rate, dot_mat);
		added_mat = add(net->hidden_weights[i], scaled_mat);
		matrix_free(net->hidden_weights[i]); // Free the old hidden_weights before replacement
		net->hidden_weights[i] = added_mat;

		 

		matrix_free(sigmoid_primed_mat);
		matrix_free(multiplied_mat);
		matrix_free(transposed_mat);
		matrix_free(dot_mat);
		matrix_free(scaled_mat);

	}

	for (int i = 0;i < hidden;i++)
	{
		matrix_free(hidden_outputs_stored[i]);
		matrix_free(hidden_errors_stored[i]);
	}
	matrix_free(output_errors);
	matrix_free(final_outputs);
	
}

void network_train_batch_imgs(NeuralNetwork* net, Img** imgs, int batch_size,int hidden) {
	for (int i = 0; i < batch_size; i++) {
		if (i % 100 == 0) printf("Img No. %d\n", i);
		Img* cur_img = imgs[i];
		Matrix* img_data = matrix_flatten(cur_img->img_data, 0); // 0 = flatten to column vector
		Matrix* output = matrix_create(10, 1);
		output->entries[cur_img->label][0] = 1; // Setting the result
		network_train(net, img_data, output,hidden);
		matrix_free(output);
		matrix_free(img_data);
	}
}

Matrix* network_predict_img(NeuralNetwork* net, Img* img) {
	Matrix* img_data = matrix_flatten(img->img_data, 0);
	Matrix* res = network_predict(net, img_data);
	matrix_free(img_data);
	return res;
}

double network_predict_imgs(NeuralNetwork* net, Img** imgs, int n) {
	int n_correct = 0;
	for (int i = 0; i < n; i++) {
		Matrix* prediction = network_predict_img(net, imgs[i]);
		if (matrix_argmax(prediction) == imgs[i]->label) {
			n_correct++;
		}
		matrix_free(prediction);
	}
	return 1.0 * n_correct / n;
}

Matrix* network_predict(NeuralNetwork* net, Matrix* input_data) {
	Matrix* hidden_inputs	= dot(net->hidden_weights, input_data);
	Matrix* hidden_outputs = apply(sigmoid, hidden_inputs);
	Matrix* final_inputs = dot(net->output_weights, hidden_outputs);
	Matrix* final_outputs = apply(sigmoid, final_inputs);
	Matrix* result = softmax(final_outputs);

	matrix_free(hidden_inputs);
	matrix_free(hidden_outputs);
	matrix_free(final_inputs);
	matrix_free(final_outputs);

	return result;
}

void network_save(NeuralNetwork* net, char* file_string) {
	mkdir(file_string, 0777);
	// Write the descriptor file
	chdir(file_string);
	FILE* descriptor = fopen("descriptor", "w");
	fprintf(descriptor, "%d\n", net->input);
	fprintf(descriptor, "%d\n", net->hidden);
	fprintf(descriptor, "%d\n", net->output);
	fclose(descriptor);
	matrix_save(net->hidden_weights, "hidden");
	matrix_save(net->output_weights, "output");
	printf("Successfully written to '%s'\n", file_string);
	chdir("-"); // Go back to the orignal directory
}

NeuralNetwork* network_load(char* file_string) {
	NeuralNetwork* net = malloc(sizeof(NeuralNetwork));
	char entry[MAXCHAR];
	chdir(file_string);

	FILE* descriptor = fopen("descriptor", "r");
	fgets(entry, MAXCHAR, descriptor);
	net->input = atoi(entry);
	fgets(entry, MAXCHAR, descriptor);
	net->hidden = atoi(entry);
	fgets(entry, MAXCHAR, descriptor);
	net->output = atoi(entry);
	fclose(descriptor);
	net->hidden_weights = matrix_load("hidden");
	net->output_weights = matrix_load("output");
	printf("Successfully loaded network from '%s'\n", file_string);
	chdir("-"); // Go back to the original directory
	return net;
}

void network_print(NeuralNetwork* net) {
	printf("# of Inputs: %d\n", net->input);
	printf("# of Hidden: %d\n", net->hidden);
	printf("# of Output: %d\n", net->output);
	printf("Hidden Weights: \n");
	matrix_print(net->hidden_weights);
	printf("Output Weights: \n");
	matrix_print(net->output_weights);
}

void network_free(NeuralNetwork *net) {
	matrix_free(net->hidden_weights);
	matrix_free(net->output_weights);
	free(net);
	net = NULL;
}