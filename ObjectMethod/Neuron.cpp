#include "Neuron.h"
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <unistd.h>

Neuron::~Neuron()
{
  delete[] weights;
}
//->
void Neuron::calc_output(int prev_layer_size, float* input)
{
  float somme = somme_input(prev_layer_size,input)+bias;
  output = modifier(somme);
  delete []input;
}
float Neuron::somme_input(int prev_layer_size, float* input)
{
  float som = 0;
  for(int i = 0;i < prev_layer_size;i++)
  {
    som += input[i]*weights[i];
  }
  return som;
}
void Neuron::Init_weights(int prev_size)
{
  weights = new float[prev_size];
  for(int i=0; i < prev_size;i++)
  {
    weights[i] = 2*((float)rand()/RAND_MAX)-1;
  }
}
//<-
void Neuron::back_propagation_body(Neuron* neurons, float* input, int next_l_size, int prev_size, int index)
{
  float som = 0;
  float* weights_n;
  float error_n;
  for(int i = 0;i < next_l_size;i++)
  {
    weights_n = neurons[i].get_weights();
    error_n = neurons[i].get_error();
    som += weights_n[index]*error_n;
  }
  error = som*modifier_d(output);
  bias -= LEARNING_RATE*error;
  for(int i = 0;i < prev_size; i++)
  {
    weights[i] -= LEARNING_RATE*error*input[i];
  }
  delete []input;
}
void Neuron::back_propagation_head(float* training_outputs, float* prev_outputs, int index, int prev_size)
{
  error = output-training_outputs[index];
  bias -= LEARNING_RATE*error;
  for(int i = 0;i < prev_size;i++)
  {
    weights[i] -= LEARNING_RATE*error*prev_outputs[i];
  }
  delete []prev_outputs;
}
//get
float Neuron::get_output(){return output;}
float Neuron::get_bias(){return bias;}
float* Neuron::get_weights(){return weights;}
float Neuron::get_error(){return error;}
//set
void Neuron::set_output(float outp){output=outp;}
void Neuron::set_modifiers(float (*m)(float), float (*m_d)(float))
{
  modifier = m;
  modifier_d = m_d;
}
void Neuron::set_bias(float f){bias=f;}
void Neuron::print_weights(int prev_size)
{
  for(int i=0; i < prev_size;i++)
  {
    std::cout <<"i: "<<weights[i]<<std::endl;
  }
}
