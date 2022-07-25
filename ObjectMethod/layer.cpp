#include <stdlib.h>
#include <time.h>
#include "layer.h"
#include <iostream>
//constructor
void layer::gen_layer(int index, int n,float (*m)(float),float (*m_d)(float))
{
  size = n;
  output = new float[n];
  layer_neurons = new Neuron[n];
  if(index > 0)
  {
    for(int i = 0; i < n; i++)
    {
      layer_neurons[i].set_modifiers(m,m_d);
      layer_neurons[i].Init_weights(prev_layer->get_size());
    }
  }
}
layer::~layer()
{
  delete[] output;
  delete[] layer_neurons;
}
//->
void layer::calc_outputs()
{
  for(int i = 0;i < size;i++)
  {
    layer_neurons[i].calc_output(prev_layer->get_size(),get_input());
    output[i] = layer_neurons[i].get_output();
  }
  if(next_layer != NULL)
    next_layer->calc_outputs();
}
//<-
void layer::back_propagation(int data_index, layer* f_layer, float* training_output = NULL)
{
  //std::cout << "            layer:"<<std::endl;
  if(next_layer != NULL)
  {
    for(int i = 0;i < size;i++)
    {
      layer_neurons[i].back_propagation_body(next_layer->get_neurons(),
					     prev_layer->get_output(),
					     next_layer->get_size(),
					     prev_layer->get_size(),i);
    }
  }
  else
  {
    for(int i = 0;i < size;i++)
    {
      layer_neurons[i].back_propagation_head(training_output,
					     prev_layer->get_output(),
					     data_index,prev_layer->get_size());
    }
  }
  if(prev_layer != f_layer)
    prev_layer->back_propagation(data_index,f_layer);
}
//get
float* layer::get_output()
{
  float* output_ = new float[size];
  for(int i = 0; i < size;i++)
  {
    output_[i] = output[i];
  }
  return output_;
}
int layer::get_size(){return size;}
float* layer::get_input()
{
  return prev_layer->get_output();
}
Neuron* layer::get_neurons()
{
  return layer_neurons;
}
//set
void layer::set_output(float* outp){output=outp;}
void layer::set_connections(layer* prev,layer* next)
{
  prev_layer = prev;
  next_layer = next;
}
//
int layer::get_neuron_index(Neuron* n)
{
  int i;
  for(int j = 0;j < size;j++)
  {
    if(&layer_neurons[j] == n)
      i = j;
  }
  return i;
}
Neuron* layer::get_neuron(int index)
{
  return &layer_neurons[index];
}
