#ifndef NNLAYER
#define NNLAYER
#include "Neuron.h"
class layer
{
 private:
  //layer data
  int size;
  float* output;
  Neuron* layer_neurons;
 public:
  //external data
  layer* prev_layer;
  layer* next_layer;
  //constructor
  void gen_layer(int,int,float (*)(float),float (*)(float));
  ~layer();
  //methods
  //->
  void calc_outputs();
  //<-
  void back_propagation(int,layer*,float*);
  //get
  float* get_output();
  int get_size();
  float* get_input();
  Neuron* get_neurons();
  //set
  void set_output(float*);
  void set_connections(layer*,layer*);
  //
  int get_neuron_index(Neuron*);
  Neuron* get_neuron(int);
};
#endif
