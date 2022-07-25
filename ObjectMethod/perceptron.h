#ifndef PERCEPTRON
#define PERCEPTRON
#define RAND NULL

class perceptron
{
 private:
  int training_inputs[4][3] = {{0,0,1},
                               {1,1,1},
                               {1,0,1},
                               {0,1,1}};
  int training_outputs[4] = {0,1,1,0};
  float weights[3];
  float* outputs;
 public:
  void init_w();
  void learn(int);
  float test(int*);
  float* neural_input_somme(int);
  float* act_f_calc(float*,int ,float (*)(float));
  float* error_calc(int);
  float* adjustment_calc(float*,int,float (*)(float));
  void resize_weight(float*,int);
  void print(int);
  void print_w();
  void print_er(float*,int);
};
#endif
