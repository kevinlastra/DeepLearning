#include "NeuralNetwork.h"
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <math.h>
#include <fstream>
#include <iomanip>

NeuralNetwork::NeuralNetwork(int d_s, int nn_size, int* sl)
  :NN_size(nn_size), size_of_layers(sl), data_size(d_s)
{
  srand(time(NULL));
  InitInput();

  outputs = new float*[NN_size-1];
  weights = new float**[NN_size-1];
  error = new float*[NN_size-1];
  bias = new float*[NN_size-1];
  for(int i = 0; i < NN_size-1; i++)
  {
    outputs[i] = new float[size_of_layers[i+1]];
    
    for(int j = 0; j < size_of_layers[i+1]; j++)
      outputs[i][j] = 0;
    
    weights[i] = new float*[size_of_layers[i]];
    bias[i] = new float[size_of_layers[i]];
    error[i] = new float[size_of_layers[i]];
    for(int j = 0; j < size_of_layers[i]; j++)
    {
      weights[i][j] = new float[size_of_layers[i+1]];
      for(int h = 0; h < size_of_layers[i+1]; h++)
	weights[i][j][h] = 2*((float)rand()/RAND_MAX)-1;
      bias[i][j] = 0;
      error[i][j] = 0;
    }
  }
}
void NeuralNetwork::InitInput()
{
  input = new float*[data_size];
  t_output = new float*[data_size];
  for(int i = 0; i < data_size; i++)
  {
    input[i] = new float[size_of_layers[0]];
    for(int j = 0; j < size_of_layers[0]; j++)
      input[i][j] = (rand()%10);
    t_output[i] = new float[size_of_layers[NN_size-1]];
    for(int j = 0; j < size_of_layers[NN_size-1]; j++)
      t_output[i][j] = (input[i][0] > 2 && input[i][0] < 9 && input[i][1] > 2 && input[i][1] < 9)? 1: 0;
			 
  }
}
void NeuralNetwork::learn()
{
  for(int i = 0; i < data_size; i++)
  {
    Calc_front(i);
    Calc_back(i);
  }
}
void NeuralNetwork::Calc_front(int input_index)
{
  float* M;
  for(int i = 0; i < size_of_layers[1]; i++)
  {
    M = MxMi(input[input_index], weights[0], i, size_of_layers[0]);
    outputs[0][i] = sigmoid(M[i] + bias[0][i]);
    delete []M;
  }
  
  for(int i = 1; i < NN_size-1; i++)
  {
    for(int h = 0; h < size_of_layers[i+1]; h++)
    {
      M = MxMi(outputs[i-1], weights[i], h, size_of_layers[i]);
      outputs[i][h] = sigmoid(M[h] + bias[i][h]); 
    }
    delete []M;
  }
}
void NeuralNetwork::Calc_back(int input_index)
{
  for(int i = 0; i < size_of_layers[NN_size-1]; i++)
  {
    error[NN_size-2][i] = outputs[NN_size-2][i]-t_output[input_index][i];
    bias[NN_size-2][i] -= LEARNING_RATE*error[NN_size-2][i];
    
  }
  for(int j = 0; j < size_of_layers[NN_size-2]; j++)
  {
    for(int h = 0; h < size_of_layers[NN_size-1]; h++)
    {
      weights[NN_size-2][j][h] -= LEARNING_RATE*error[NN_size-2][h]*outputs[NN_size-3][j];
    }
  }
  
  float som;
  for(int i = NN_size-2; i > 0; i--)
  {
    for(int j = 0; j < size_of_layers[i]; j++)
    {
      for(int h = 0; h < size_of_layers[i+1]; h++)
      {
	som += weights[i][j][h]*error[i][h];
      }
      
      error[i-1][j] = som*sigmoid_d(outputs[i-1][j]);
      bias[i-1][j] -= LEARNING_RATE*error[i-1][j];
    }
    for(int j = 0; j < size_of_layers[i-1]; j++)
    {
      for(int h = 0; h < size_of_layers[i]; h++)
      {
	if(i-1 == 0)
	  weights[i-1][j][h] -= LEARNING_RATE*error[i-1][h]*input[input_index][j];
	else
	  weights[i-1][j][h] -= LEARNING_RATE*error[i-1][h]*outputs[i-1][j];
      }
    }
  } 
}
void NeuralNetwork::test_data()
{
  float* test = new float[size_of_layers[0]];
  std::cout << "Testing data" << std::endl;
  std::cout << "Complete the inputs writing values between [0,9]" << std::endl;
  for(int i = 0; i < size_of_layers[0]; i++)
  {
    std::cout << "Input_"<<i<<": ";
    std::cin >> test[i];
  }
  float* M;
  for(int i = 0; i < size_of_layers[1]; i++)
  {
    M = MxMi(test, weights[0], i, size_of_layers[0]);
    outputs[0][i] = sigmoid(M[i] + bias[0][i]);
    delete []M;
  }
  
  for(int i = 1; i < NN_size-1; i++)
  {
    for(int h = 0; h < size_of_layers[i+1]; h++)
    {
      M = MxMi(outputs[i-1], weights[i], h, size_of_layers[i]);
      outputs[i][h] = sigmoid(M[h] + bias[i][h]); 
    }
    delete []M;
  }
  std::cout << "Output" << std::endl << std::endl;
  for(int i = 0; i < size_of_layers[NN_size-1]; i++)
  {
    std::cout << i << ": " << outputs[NN_size-2][i] << std::endl;
  }
}
float* NeuralNetwork::MxC(float* M, float x, int M_size)
{
  float* M_ = new float[M_size];
  for(int i = 0; i < M_size; i++)
  {
    M_[i] = M[i]*x;
  }
  return M_;
}
float* NeuralNetwork::MxM(float* M1, float* M2, int M_size)
{
  float* M_ = new float[M_size];
  for(int i = 0; i < M_size; i++)
  {
    M_[i] = M1[i]*M2[i]; 
  }
  return M_;
}
float* NeuralNetwork::MxMi(float* M2, float** M1, int index, int size)
{
  float* som = new float[size];
  for(int i = 0; i < size; i++)
  {
    som[i] = M1[i][index] * M2[i];
  }
  return som;
}
float NeuralNetwork::sigmoid(float x)
{
  return 1/(1+exp(-x));
}
float NeuralNetwork::sigmoid_d(float x)
{
  return x*(1-x);
}
void NeuralNetwork::print_IO()
{
  std::cout << "Printing data test:" << std::endl;
  std::cout << "    In0 In1 Out0"<<std::endl;
  for(int i = 0; i < data_size; i++)
  {
    std::cout << i << ":   ";
    for(int j = 0; j < size_of_layers[0]; j++)
      std::cout << input[i][j] << "   " ;
    
    for(int j = 0; j < size_of_layers[NN_size-1]; j++)
      std::cout << t_output[i][j] << std::endl;
  }
}
void NeuralNetwork::dot_data()
{
  std::ofstream file("Network.dot");
  file << std::setprecision(2);
  file << "digraph { graph[pad=\"0.5\", nodesep=\"1.5\"]; splines = false;" << std::endl;
  for(int i = 0; i < NN_size; i++)
  {
    for(int j = 0; j < size_of_layers[i]; j++)
    {
      if(i == 0)
	file << "a_" << i << j
	     << "[shape=circle, label=\"a_"<< i << j <<"="<< input[data_size-1][j] <<"\"];"
	     <<std::endl;
      else
	file << "a_" << i << j << "[shape=circle, label=\"a_"<< i << j <<"="<< outputs[i-1][j]
	     <<	"\ne: " << error[i-1][j] <<"\"];" <<std::endl;
    }
  }
  for(int i = 0; i < NN_size-1; i++)
  {
    for(int j = 0; j < size_of_layers[i]; j++)
    {
      for(int h = 0; h < size_of_layers[i+1]; h++)
      {
	file << "a_" << i << j << "->" << "a_" << i+1 << h << " [label = \"w_"<< i << j <<"= "<< weights[i][j][h]<<"\"];" <<std::endl;
      }
    }
  }
  file << "}" << std::endl;
  file.close();

  system("dot -Tps Network.dot -o outfile.ps");
  system("display Network.dot &");
}
