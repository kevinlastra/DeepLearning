#include "perceptron.h"
#include "mathF.h"
#include <stdlib.h>
#include <time.h>
#include <iostream>

void perceptron::init_w()
{
  srand(time(RAND));
  
  for(int i=0; i < 3;i++)
  {
    weights[i] = 2*((float)rand()/RAND_MAX)-1;
  }
}
float* perceptron::neural_input_somme(int n)
{
  float* som = new float[n];
  for(int i = 0;i < n;i++)
  {
    for(int j = 0;j < 3;j++)
    {
      som[i] += training_inputs[i][j]*weights[j];
    }
  }
  return som;
}
float* perceptron::act_f_calc(float* inputs,int n, float (*func)(float))
{
  
  float* som = new float[n];
  for(int i = 0;i < n;i++)
  {
    som[i] = func(inputs[i]);
  } 
  return som;
}
float* perceptron::error_calc(int n)
{
  float* som = new float[n];
  for(int i = 0;i < n;i++)
  {
    som[i] = training_outputs[i]-outputs[i];
  }
  return som;
}
float* perceptron::adjustment_calc(float* error,int n, float (*func)(float))
{
  float* som = new float[n];
  for(int i = 0;i < n;i++)
  {
    som[i] = error[i]*func(outputs[i]);
  }
  return som;
}
void perceptron::resize_weight(float* adj,int n)
{
  float som = 0;
  for(int i = 0;i < 3;i++)
  {
    for(int j = 0; j < n;j++)
    {
      som += training_inputs[j][i]*adj[j];
    }
    weights[i] += som;
    som = 0;
  }
}
void perceptron::learn(int epochs)
{
  
  float* error;
  float* adj;
  int n = 5;
  init_w();
  
  for(int i = 0;i < epochs;i++)
  {
    outputs = act_f_calc(neural_input_somme(n),n,sigmoid);
    error = error_calc(n);
    adj = adjustment_calc(error,n,sigmoid_d);
    resize_weight(adj,n);
  }
}
float perceptron::test(int* t)
{
  float som = 0;
  for(int i = 0; i < 3;i++)
  {
    som += t[i]*weights[i];
  }
  return sigmoid(som);
}
void perceptron::print(int n)
{
  std::cout <<"OUTPUT:"<<std::endl;
  for(int i = 0;i <n;i++)
  {
    std::cout <<"  "<<outputs[i];
  }
  std::cout<<std::endl;
}
void perceptron::print_w()
{
  std::cout <<"WEIGHTs:"<<std::endl;
  for(int i = 0;i < 3;i++)
  {
    std::cout <<"  "<<weights[i];
  }
  std::cout<<std::endl;
}
void perceptron::print_er(float* error,int n)
{
  std::cout <<"ERRORs:"<<std::endl;
  for(int i = 0;i < n;i++)
  {
    std::cout <<"  "<<error[i];
  }
  std::cout<<std::endl;
}
