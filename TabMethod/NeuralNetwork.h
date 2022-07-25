#ifndef NN
#define NN


class NeuralNetwork
{
 private:
  int NN_size;
  int* size_of_layers;
  int data_size;

  float** input;
  float** t_output;
  
  float** outputs;
  float*** weights;
  float** error;
  float** bias;

  const float LEARNING_RATE = 0.0001;
 public:
  NeuralNetwork(int,int,int*);
  void InitInput();

  void learn();
  void Calc_front(int);
  void Calc_back(int);
  void test_data();
  
  float* MxC(float*, float, int);
  float* MxM(float*, float*, int);
  float* MxMi(float*, float**, int, int);
  float sigmoid(float);
  float sigmoid_d(float);

  void print_IO();
  void dot_data();
};
#endif
