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

int flag1 = 0;
int flag2 = 0;
int flag3 = 0;

int pipeline_start_unit1 = 0;
int pipeline_start_unit2 = 1;
int pipeline_start_unit3 = 1;

int counter_1 = 0;
int counter_2 = 0;
int counter_3 = 0;

//boolean variables
int last_stage_first_backProp = 0;
int done_firstback_prop = 0;
int gonnaDoBackProp = 0;

//boolean variables
int stage1fp = 1;
int stage1bp = 0;

//boolean variables
int stage2fp = 1;
int stage2bp = 0;

//boolean variables
int stage3fp = 1;
int stage3bp = 0;

int backPropStart = 0;
int num_back_prop = 0;
int num_fwd_prop = 0;

//counters back prop
int num_back_prop_stage1 = 0;
int num_back_prop_stage2 = 0;
int num_back_prop_stage3 = 0;

//
int fwddirection = 0;

int num_stages = 3;

pthread_mutex_t mutex;


int number_imgs = 10000;


void myThreadFun(int id)
{
	printf("%s", "In thread");
	printf("%d", id);
}

int workFunc(int x)
{
	
	if (x == 1)
	{
		
		if (backPropStart == 1 && num_back_prop != num_stages - 1)
		{
			//do nothing
		}
		
		else
		{
			if (backPropStart != 1)
			{
				Img* cur_img = imgs[0];
				Matrix* img_data = matrix_flatten(cur_img->img_data, 0); // 0 = flatten to column vector
				Matrix* output = matrix_create(10, 1);
				output->entries[cur_img->label][0] = 1; // Setting the result
				network_train(net, img_data, output, 1, 0);
				matrix_free(output);
				matrix_free(img_data);
			}
			else {
				if (num_back_prop == num_stages - 2)
				{

					num_back_prop_stage1++;
					//then simply backpropagate

				}
				else
				{
					//alternate between forward and backward
					if (stage1fp == 1)
					{
						//go forward;
						stage2fp = 0;
					}
					else
					{
						//go backward
						num_back_prop_stage1++;
						stage2fp = 1;
					}


				}
			}
			
		}
			counter_1++;
		printf("done training Stage 1\n");
	}

	else if (x == 2)
	{
		/*Img* cur_img = imgs[1];
		Matrix* img_data = matrix_flatten(cur_img->img_data, 0); // 0 = flatten to column vector
		Matrix* output = matrix_create(10, 1);
		output->entries[cur_img->label][0] = 1; // Setting the result
		network_train(net, img_data, output);
		matrix_free(output);
		matrix_free(img_data);*/

		if (backPropStart == 1 && num_back_prop < num_stages - 1)
		{
			//do nothing
		}

		else
		{

			if (backPropStart != 1)
			{

				if (pipeline_start_unit2 == 1)
				{
					counter_2++;
					if (counter_2 > 0)
					{
						pipeline_start_unit2 = 0;
					}

					
				}

				else
				{
					counter_2++;
					/*if (last_stage_first_backProp == 0)
					{
						Img* cur_img = imgs[0];
						Matrix* img_data = matrix_flatten(cur_img->img_data, 0); // 0 = flatten to column vector
						Matrix* output = matrix_create(10, 1);
						output->entries[cur_img->label][0] = 1; // Setting the result
						network_train(net, img_data, output, 2, 0);
						matrix_free(output);
						matrix_free(img_data);
					}*/
					//do nothing
				}

			}
			else
			{
				if (num_back_prop == num_stages - 2)
				{

					num_back_prop_stage2++;
					//then simply backpropagate

				}
				else
				{
					//alternate between forward and backward
					if (stage2fp == 1)
					{
						//go forward;
						stage2fp = 0;
					}
					else
					{
						//go backward
						num_back_prop_stage1++;
						stage2fp = 1;
					}
					

				}

			}

		}
		printf("done training Stage 2\n");
	}
	
	else if (x == 3)
	{
		/*Img* cur_img = imgs[2];
		Matrix* img_data = matrix_flatten(cur_img->img_data, 0); // 0 = flatten to column vector
		Matrix* output = matrix_create(10, 1);
		output->entries[cur_img->label][0] = 1; // Setting the result
		network_train(net, img_data, output);
		matrix_free(output);
		//matrix_free(img_data);
		printf("done training Image 3");
		pthread_mutex_lock(&mutex);
		flag3 = 1;
		pthread_mutex_unlock(&mutex); */
		if (backPropStart == 1 && num_back_prop != num_stages - 3)
		{
			//do nothing
		}
		else
		{
			if (backPropStart != 1)
			{
				if (pipeline_start_unit3 == 1)
				{
					counter_3++;
					if (counter_3 > 1)
					{
						pipeline_start_unit3 = 0;
					}

					/*pthread_mutex_lock(&mutex);
					flag3 = 1;
					pthread_mutex_unlock(&mutex);

					while (1 < 2)
					{
						pthread_mutex_lock(&mutex);
						if (flag1 == 1 && flag2 == 1 && flag3 == 1)
						{

							pthread_mutex_unlock(&mutex);
							break;

						}
						pthread_mutex_unlock(&mutex);
					}*/
				}
				
				else
				{
					counter_3++;
					/*Img* cur_img = imgs[0];
					Matrix* img_data = matrix_flatten(cur_img->img_data, 0); // 0 = flatten to column vector
					Matrix* output = matrix_create(10, 1);
					output->entries[cur_img->label][0] = 1; // Setting the result
					network_train(net, img_data, output, 3, 0);
					matrix_free(output);
					matrix_free(img_data);*/

					//do nothing

				}



				
			}
			else
			{
				//backpropagate and alternateforward back to 1
				if (num_back_prop == num_stages - 2)
				{

					num_back_prop_stage1++;
					//then simply backpropagate

				}
				else
				{
					//alternate between forward and backward
					if (stage3fp == 1)
					{
						//go forward;
					}
					else
					{
						//go backward
						stage3fp = 0;
						num_back_prop_stage1++;
					}


				}


			}
		}
	
			printf("done training Stage 3\n");
	}

		
	
	return 0;
}


int main()
{
	srand(time(NULL));

	//TRAINING
	 number_imgs = 10000;
	 imgs = csv_to_imgs("./MNIST_data_train.csv", number_imgs);
	 net = network_create(784, 300,200,100,3, 10, 0.1);
	 clock_t starttime = clock() / (CLOCKS_PER_SEC / 1000);
	 time_t start , end;
	 time(&start);
	 //network_train_batch_imgs(net, imgs, number_imgs);
	 time(&end);
	 printf("%f", (double)(end-start));
	 clock_t endtime = clock() / (CLOCKS_PER_SEC / 1000);
	 network_save(net, "testing_net");

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

	 /*thread t1(workFunc, 1);
	 thread t2(workFunc, 2);
	 thread t3(workFunc, 3);*/

	 pthread_t thread_id1, thread_id2, thread_id3;

	 int thread_run_forward_count = 0;
	 int thread_run_count = 0;
	 int num_stages = 3;

	 while (thread_run_count < 4)
	 {
		 //creating 3 threads which represent pipeline stages
		 pthread_create(&thread_id1, NULL, workFunc, 1);
		 pthread_create(&thread_id2, NULL, workFunc, 2);
		 pthread_create(&thread_id3, NULL, workFunc, 3);

		 pthread_join(thread_id1, NULL);
		 pthread_join(thread_id2, NULL);
		 pthread_join(thread_id3, NULL);
		 printf("Done Run\n");

		 thread_run_count++;
		 thread_run_forward_count++;

		 if (backPropStart == 0)
		 {
			 if (num_fwd_prop == num_stages)
			 {
				 backPropStart = 1;
				 num_fwd_prop = 0;
			 }
			 else
			 {
				 num_fwd_prop++;
			 }
		 }
		 else
		 {
			 if (num_fwd_prop == num_stages)
			 {
				 backPropStart = 1;
				 num_fwd_prop = 0;
			 }
			 else
			 {
				 num_fwd_prop++;
			 }

		 }

		 

	 }
	 //printf("Done both the threads");

	imgs_free(imgs, number_imgs);
	network_free(net);
	//printf("%d", (endtime - starttime));
	//int r = 0;
	//scanf("%d", &r);
	getch();
	return 0;
}
