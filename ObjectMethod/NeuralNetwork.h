#ifndef NNETWORK
#define NNETWORK
#include "layer.h"
#include "DataManager.h"
#include <string>

class NeuralNetwork
{
 private:
  float** training_inputs;
  float* training_outputs;
  //structure
  int structure[3] = {2,4,1};
  int size = 3;
  //
  layer* f_layer;
  layer* l_layer;
 public:
  void build();
  ~NeuralNetwork();
  void training_data();
  bool verif_output_data(float*);
  void learn(int,int);
  void test(float*);
  void save_neural_set();
  void load_neural_set();
  void Normalize_input();
  void print_output(float**);
  void print_data_set();
  void Graph_data_training(float**,float*,std::string);
  void Graph_data_output(float**,float**,std::string);
};
#endif
