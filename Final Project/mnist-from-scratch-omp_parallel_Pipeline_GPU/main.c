#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "util/img.h"
#include "neural/activations.h"
#include "neural/nn.h"
#include "matrix/matrix.h"
#include "matrix/ops.h"
#include <pthread.h>

Img** imgs = NULL;
NeuralNetwork* net = NULL;
Matrix* output = NULL;
int num_stages = 12;

int flag1 = 0;
int flag2 = 0;
int flag3 = 0;

int* pipeline_start_unit;

int* counter;
//boolean variables
int last_stage_first_backProp = 0;
int done_firstback_prop = 0;
int gonnaDoBackProp = 0;

//boolean variables
int* stagefp;
int* stagebp;

int backPropStart = 0;
int num_back_prop = 0;
int num_fwd_prop = 0;

//counters back prop

int* num_back_prop_stage;




//counter forward prop

int* num_fwd_prop_stage;

//
int fwddirection = 0;



pthread_mutex_t mutex;

int num_batches_trained = 0;
int number_imgs = 10000;


void reinitializeVariables()
{
	flag1 = 0;
	flag2 = 0;
	flag3 = 0;

	
	pipeline_start_unit = malloc(sizeof(int) * num_stages);
	
	pipeline_start_unit[0] = 1;
	for (int i = 1;i < num_stages;i++)
	{
		pipeline_start_unit[i] = 0;
	}
	

	counter = malloc(sizeof(int) * num_stages);
	for (int i = 1;i < num_stages;i++)
	{
		counter[i] = 0;
	}


	//boolean variables
	last_stage_first_backProp = 0;
	done_firstback_prop = 0;
	gonnaDoBackProp = 0;

	//boolean variables
	
	stagefp = malloc(sizeof(int)*num_stages);
	stagebp = malloc(sizeof(int) * num_stages);
	for (int i = 0;i < num_stages;i++)
	{
		stagefp[i] = 1;
		stagebp[i] = 0;
	}

	backPropStart = 0;
	num_back_prop = 0;
	num_fwd_prop = 0;

	//counters back prop
	num_back_prop_stage = malloc(sizeof(int)*num_stages);
	//counter forward prop
	num_fwd_prop_stage = malloc(sizeof(int)*num_stages);
	for (int i = 0;i < num_stages;i++)
	{
		num_back_prop_stage[i] = 0;
		num_fwd_prop_stage[i] = 0;
	}
	
	fwddirection = 0;
}

void myThreadFun(int id)
{
	printf("%s", "In thread");
	printf("%d", id);
}

int workFunc(int stage)
{
	
		if (backPropStart == 1 && num_back_prop < num_stages - stage)
		{
			//do nothing
		}
		
		else
		{
			if (backPropStart != 1)
			{
				
				//add here

				if (pipeline_start_unit[stage-1] == 0)
				{
					if (stage == 4)
					{
						
						int u = counter[stage - 1];
						int dbg = 1;
					}
					
					counter[stage - 1]++;
					if (counter[stage-1] > stage-2)
					{
						pipeline_start_unit[stage-1] = 1;
					}
					
				}

				else
				{
					counter[stage-1]++;
					Matrix* img_data = NULL;
					if (stage == 1)
					{
						//img_data = NULL;
						Img* cur_img = imgs[(num_batches_trained * num_stages) + num_fwd_prop_stage[stage - 1]];
						img_data = matrix_flatten(cur_img->img_data, 0); // 0 = flatten to column vector
					}
					
					network_train(net, img_data, output, stage, num_fwd_prop_stage[stage-1],num_stages);
					num_fwd_prop_stage[stage-1]++;
					//do nothing
				}


			}
			else
			{
				if (num_back_prop == num_stages - stage)
				{

					
					network_backprop(net, output, stage, num_back_prop_stage[stage-1],num_stages);
					num_back_prop_stage[stage-1]++;
					//then simply backpropagate

				}
				else
				{
					//alternate between forward and backward
					if (stagefp[stage-1] == 1)
					{
						//go forward;
						stagefp[stage-1] = 0;
						if (num_fwd_prop_stage[stage-1] != num_stages)
						{
							Img* cur_img = imgs[(num_batches_trained * num_stages)+num_fwd_prop_stage[stage-1]];
							Matrix* img_data = matrix_flatten(cur_img->img_data, 0); // 0 = flatten to column vector
							output->entries[cur_img->label][0] = 1; // Setting the result
							network_train(net, img_data, output, stage, num_fwd_prop_stage[stage-1],num_stages);
							matrix_free(img_data);
							num_fwd_prop_stage[stage-1]++;
						}
					}
					else
					{
						//go backward
						if (num_back_prop_stage[stage-1] != num_stages)
						{
							network_backprop(net, output, stage, num_back_prop_stage[stage-1],num_stages);
							num_back_prop_stage[stage-1]++;
						}
						stagefp[stage-1] = 1;
					}

				}
			}
			
		}
				

	return 0;
}


int main()
{
	srand(time(NULL));

	//TRAINING
	 number_imgs = 1000;
	 imgs = csv_to_imgs("./MNIST_data_train.csv", number_imgs);
	 net = network_create(784,10, 0.1, num_stages);
	 clock_t starttime = clock() / (CLOCKS_PER_SEC / 1000);
	 time_t start , end;
	 time(&start);
	 clock_t t;
	 t = clock();
	 
	 output = matrix_create(10, 1);

	 if (pthread_mutex_init(&mutex, NULL) != 0) {
		 printf("\n mutex init has failed\n");
		 return 1;
	 }

	// PREDICTING
	/*int number_imgs = 3000;
	Img** imgs = csv_to_imgs("./mnist_test.csv", number_imgs);
	NeuralNetwork* net = network_load("testing_net");
	double score = network_predict_imgs(net, imgs, 1000);
	printf("Score: %1.5f\n", score);
	*/

	 pthread_t thread_id1, thread_id2, thread_id3, thread_id4, thread_id5, thread_id6, thread_id7, thread_id8;

	 pthread_t pt[8];
	 int thread_run_forward_count = 0;
	 int thread_run_count = 0;
	 

	 num_batches_trained = 0;
	 int batch_counter = 0;
	 
	 while (num_batches_trained < (int)(1000/num_stages))
	 {
		 
		 reinitializeVariables();
		 while (1 != 0)
		 {
			 //creating 3 threads which represent pipeline stages
			 for (int i = 0;i < num_stages;i++)
			 {
				 pthread_create(&pt[i], NULL, workFunc, i+1);
			 }
			 
			 /*pthread_create(&thread_id1, NULL, workFunc, 1);
			 pthread_create(&thread_id2, NULL, workFunc, 2);
			 pthread_create(&thread_id3, NULL, workFunc, 3);
			 pthread_create(&thread_id4, NULL, workFunc, 4);
			 pthread_create(&thread_id5, NULL, workFunc, 5);
			 pthread_create(&thread_id6, NULL, workFunc, 6);
			 pthread_create(&thread_id7, NULL, workFunc, 7);
			 pthread_create(&thread_id8, NULL, workFunc, 8);*/

			 for (int i = 0;i < num_stages;i++)
			 {
				 pthread_join(pt[i], NULL);

				 /*pthread_join(thread_id2, NULL);
				 pthread_join(thread_id3, NULL);
				 pthread_join(thread_id4, NULL);
				 pthread_join(thread_id5, NULL);
				 pthread_join(thread_id6, NULL);
				 pthread_join(thread_id7, NULL);
				 pthread_join(thread_id8, NULL);*/

			 }
			 //printf("Done Run\n");

			 
			 if (backPropStart == 0)
			 {
				 num_fwd_prop++;
				 if (num_fwd_prop == num_stages)
				 {
					 backPropStart = 1;
					 //num_fwd_prop = 0;
				 }
			 }
			 else
			 {
				 num_back_prop++;
			 }

			 /*if (num_back_prop_stage3 == 3 && num_back_prop_stage2 == 2 && num_back_prop_stage1 == 2)
			 {
				 int dbg = 1;
			 }*/
			 /*printf("Fwd prop of stage 1\n");
			 printf("%d\n", num_fwd_prop_stage[0]);
			 printf("Fwd prop of stage 2\n");
			 printf("%d\n", num_fwd_prop_stage[1]);
			 printf("Fwd prop of stage 3\n");
			 printf("%d\n", num_fwd_prop_stage[2]);
			 printf("Fwd prop of stage 4\n");
			 printf("%d\n", num_fwd_prop_stage[3]);

			 printf("Back prop of stage 4\n");
			 printf("%d\n", num_back_prop_stage[3]);
			 printf("Back prop of stage 3\n");
			 printf("%d\n", num_back_prop_stage[2]);
			 printf("Back prop of stage 2\n");
			 printf("%d\n", num_back_prop_stage[1]);
			 printf("Back prop of stage 1\n");
			 printf("%d\n", num_back_prop_stage[0]);
			 printf("Stage 2 direction\n");*/
			 //printf("%d", stage2fp);
			 if (num_back_prop_stage[0] == num_stages)
			 {
				 break;
			 }
		 }


		 //printf("Done both the threads");

		 
		 //printf("%d", (endtime - starttime));
		 int r = 0;
		 //scanf("%d", &r);
		 num_batches_trained++;
		 batch_counter = batch_counter + num_stages;
		 if (batch_counter >200)
		 {
			 printf("The number of images trained is :");
			 printf("%d", batch_counter);
			 printf("\n");
			 batch_counter = 0;
		 }
	 }

	 printf("The number of images trained is :");
	 printf("%d", batch_counter);
	 printf("\n");

	 for (int i = 1;i <= num_stages;i++)
	 {
		 for (int j = 0;j < num_stages;j++)
		 {
			 
			Matrix *added_mat = add(net->hidden_weights[i-1], all_gradients[i][j]);
			//matrix_free(net->hidden_weights[i-1]); // Free the old hidden_weights before replacement
			net->hidden_weights[i-1] = added_mat;
			 
		 }
	 }


	 for (int j = 0;j < num_stages;j++)
	 {
		 Matrix * gradient = all_gradients[num_stages+1][0];
		 Matrix* added_mat = add(net->output_weights, all_gradients[num_stages+1][j]);
		 matrix_free(net->output_weights); // Free the old hidden_weights before replacement
		 net->output_weights = added_mat;

	 }
	 
	 time(&end);
	 printf("Total time taken is");
	 t = clock() - t;
	 double time_taken = ((double)t) / CLOCKS_PER_SEC; // calculate the elapsed time
	 printf("The program took %f seconds to execute", time_taken);
	 getch();
	 imgs_free(imgs, number_imgs);
	return 0;
}
