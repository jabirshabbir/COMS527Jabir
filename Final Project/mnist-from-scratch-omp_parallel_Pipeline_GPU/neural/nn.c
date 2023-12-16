#include "nn.h"
#include <sys/stat.h>
//#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "../matrix/ops.h"
#include "../neural/activations.h"

#define MAXCHAR 1000

//matix to store all activations in forward pass 
Matrix*** all_activations = NULL;
Matrix*** all_gradients = NULL;
//Matrix*** all_outputs = NULL;
Matrix*** hidden_error_list = NULL;

Matrix* activate;
int z = 0;
// 784, 300, 10
NeuralNetwork* network_create(int input, int output, double lr,int num_stages) {
	NeuralNetwork* net = malloc(sizeof(NeuralNetwork));
	net->input = input;

	Matrix **hidden_weights = malloc(sizeof(Matrix*) * num_stages);
	int* hidden_layers = malloc(sizeof(int) * (num_stages+1));
	hidden_layers[0] = input;
	hidden_layers[1] = 300;
	hidden_layers[2] = 200;
	hidden_layers[3] = 100;
	hidden_layers[4] = 50;
	hidden_layers[5] = 60;
	hidden_layers[6] = 70;
	hidden_layers[7] = 40;
	hidden_layers[8] = 80;
	hidden_layers[9] = 30;
	hidden_layers[10] = 60;
	hidden_layers[11] = 90;
	hidden_layers[12] = 120;
	hidden_layers[13] = 160;
	//hidden_layers[14] = 180;
	//hidden_layers[15] = 150;

	for (int i = 0;i < num_stages;i++)
	{
		Matrix* hidden_layer = matrix_create(hidden_layers[i+1], hidden_layers[i]);
		matrix_randomize(hidden_layer, hidden_layers[i+1]);
		hidden_weights[i] = hidden_layer;
	}
	net->hidden_weights = hidden_weights;
	
	net->output = output;
	net->learning_rate = lr;
	
	Matrix* output_layer = matrix_create(output, hidden_layers[num_stages]);

	
	matrix_randomize(output_layer, output);
	net->output_weights = output_layer;

	int num_hidden_layers = num_stages;
	
	all_activations = malloc((num_hidden_layers+2)*sizeof(Matrix**));
	all_gradients = malloc((num_hidden_layers + 2) * sizeof(Matrix**));
	hidden_error_list = malloc((num_hidden_layers + 2) * sizeof(Matrix**));

	for (int i = 0;i < num_hidden_layers + 2;i++)
	{
		all_activations[i] = malloc((num_hidden_layers + 2) * sizeof(Matrix*));
		all_gradients[i] = malloc((num_hidden_layers + 2) * sizeof(Matrix**));
		hidden_error_list[i] = malloc((num_hidden_layers + 2) * sizeof(Matrix*));
	}
	activate = NULL;
	z = 3;
	return net;
}


void network_train(NeuralNetwork* net, Matrix* input, Matrix* output, int pipeline_stage,int input_id,int num_stages)
{
	if (pipeline_stage == 1)
	{
		all_activations[pipeline_stage-1][input_id] = input;
		Matrix* hidden_inputs_1 = dot(net->hidden_weights[0], input);
		Matrix* hidden_outputs_1 = apply(sigmoid, hidden_inputs_1);
		all_activations[pipeline_stage][input_id] = hidden_outputs_1;
	}
	else if(pipeline_stage < num_stages)
	{
		Matrix* storedActivation = all_activations[pipeline_stage - 1][input_id];
		Matrix* hidden_inputs_2 = dot(net->hidden_weights[pipeline_stage -1], storedActivation);
		Matrix* hidden_outputs_2 = apply(sigmoid, hidden_inputs_2);
		all_activations[pipeline_stage][input_id] = hidden_outputs_2;
	}

	else if (pipeline_stage == num_stages)
	{
		Matrix* storedActivation = all_activations[pipeline_stage - 1][input_id];
		Matrix* hidden_inputs_4 = dot(net->hidden_weights[pipeline_stage - 1], storedActivation);
		
		
		Matrix* hidden_outputs_4 = apply(sigmoid, hidden_inputs_4);
		all_activations[pipeline_stage][input_id] = hidden_outputs_4;

		Matrix* final_inputs = dot(net->output_weights, hidden_outputs_4);
		
		//all_outputs[pipeline_stage][input_id] = final_inputs;
		Matrix* final_outputs = apply(sigmoid, final_inputs);
		all_activations[pipeline_stage+1][input_id] = final_outputs;

	}

}

void network_backprop(NeuralNetwork* net,Matrix*output,int pipeline_stage, int input_id,int num_stages)
{
	if (pipeline_stage == num_stages)
	{
		//Matrix* final_outputs = all_activations[pipeline_stage + 1][input_id];
		Matrix* final_outputs_with_activations = all_activations[pipeline_stage + 1][input_id];
		Matrix* output_errors = subtract(output, final_outputs_with_activations);
		Matrix* transposed_mat = transpose(net->output_weights);
		Matrix* hidden_errors = dot(transposed_mat, output_errors);
		matrix_free(transposed_mat);
		Matrix* sigmoid_primed_mat = sigmoidPrime(final_outputs_with_activations);
		Matrix* multiplied_mat = multiply(output_errors, sigmoid_primed_mat);
		
		Matrix* hidden_outputs = all_activations[pipeline_stage][input_id];
		transposed_mat = transpose(hidden_outputs);
		Matrix* dot_mat = dot(multiplied_mat, transposed_mat);
		//This is the gradient of output weights
		Matrix* scaled_mat = scale(net->learning_rate, dot_mat);
		all_gradients[pipeline_stage + 1][input_id] = scaled_mat;
		

		matrix_free(sigmoid_primed_mat);
		matrix_free(multiplied_mat);
		matrix_free(transposed_mat);
		matrix_free(dot_mat);
		//matrix_free(scaled_mat);

		sigmoid_primed_mat = sigmoidPrime(hidden_outputs);
		multiplied_mat = multiply(hidden_errors, sigmoid_primed_mat);
		Matrix* hidden_outputs_prev = all_activations[pipeline_stage-1][input_id];
		transposed_mat = transpose(hidden_outputs_prev);
		dot_mat = dot(multiplied_mat, transposed_mat);
		scaled_mat = scale(net->learning_rate, dot_mat);
		all_gradients[pipeline_stage][input_id] = scaled_mat;
		matrix_free(transposed_mat);

		//add specific weight here
		transposed_mat = transpose(net->hidden_weights[pipeline_stage-1]);
		Matrix* t = dot(transposed_mat, hidden_errors);
		hidden_error_list[pipeline_stage][input_id] = dot(transposed_mat, hidden_errors);
		
		matrix_free(sigmoid_primed_mat);
		matrix_free(multiplied_mat);
		matrix_free(transposed_mat);
		matrix_free(dot_mat);
		
		matrix_free(hidden_outputs);

		matrix_free(final_outputs_with_activations);
		matrix_free(output_errors);
		matrix_free(hidden_errors);

	}
	else
	{
		Matrix* hidden_outputs = all_activations[pipeline_stage][input_id];
		Matrix* hidden_errors = hidden_error_list[pipeline_stage + 1][input_id];
		Matrix* sigmoid_primed_mat = sigmoidPrime(hidden_outputs);
		Matrix* multiplied_mat = multiply(hidden_errors, sigmoid_primed_mat);
		Matrix* hidden_outputs_prev = all_activations[pipeline_stage - 1][input_id];
		Matrix* transposed_mat = transpose(hidden_outputs_prev);
		Matrix* dot_mat = dot(multiplied_mat, transposed_mat);
		Matrix* scaled_mat = scale(net->learning_rate, dot_mat);
		all_gradients[pipeline_stage][input_id] = scaled_mat;
		matrix_free(transposed_mat);

		transposed_mat = transpose(net->hidden_weights[pipeline_stage - 1]);

		hidden_error_list[pipeline_stage][input_id] = dot(transposed_mat, hidden_errors);


		matrix_free(sigmoid_primed_mat);
		matrix_free(multiplied_mat);
		matrix_free(transposed_mat);
		matrix_free(dot_mat);
		
		matrix_free(hidden_outputs);


		matrix_free(hidden_errors);

	}

	
}


void network_train_batch_imgs(NeuralNetwork* net, Img** imgs, int batch_size) {
	for (int i = 0; i < batch_size; i++) {
		if (i % 100 == 0) printf("Img No. %d\n", i);
		Img* cur_img = imgs[i];
		Matrix* img_data = matrix_flatten(cur_img->img_data, 0); // 0 = flatten to column vector
		Matrix* output = matrix_create(10, 1);
		output->entries[cur_img->label][0] = 1; // Setting the result
		network_train(net, img_data, output,1,0,4);
		matrix_free(output);
		//matrix_free(img_data);
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
	Matrix* hidden_inputs	= dot(net->hidden_weights_1, input_data);
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
	fprintf(descriptor, "%d\n", net->hidden1);
	fprintf(descriptor, "%d\n", net->output);
	fclose(descriptor);
	matrix_save(net->hidden_weights_1, "hidden1");
	matrix_save(net->hidden_weights_2, "hidden2");
	matrix_save(net->hidden_weights_3, "hidden3");
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
	net->hidden1 = atoi(entry);
	fgets(entry, MAXCHAR, descriptor);
	net->output = atoi(entry);
	fclose(descriptor);
	net->hidden_weights_1 = matrix_load("hidden");
	net->output_weights = matrix_load("output");
	printf("Successfully loaded network from '%s'\n", file_string);
	chdir("-"); // Go back to the original directory
	return net;
}

void network_print(NeuralNetwork* net) {
	printf("# of Inputs: %d\n", net->input);
	printf("# of Hidden: %d\n", net->hidden1);
	printf("# of Output: %d\n", net->output);
	printf("Hidden Weights: \n");
	matrix_print(net->hidden_weights_1);
	printf("Output Weights: \n");
	matrix_print(net->output_weights);
}

void network_free(NeuralNetwork *net, int num_stages)
{
	
	matrix_free(net->output_weights);
	for (int i = 0;i < num_stages;i++)
	{
		matrix_free(net->hidden_weights[i]);
	}
	
	free(net);
	net = NULL;
}