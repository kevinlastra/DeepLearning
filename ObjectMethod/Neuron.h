#ifndef NEURON
#define NEURON

#define LEARNING_RATE 0.00001
class Neuron
{
 private:
  //neuron data
  float* weights;
  float bias=0;
  float (*modifier)(float);
  float (*modifier_d)(float);
  float output=0;
  float error=0;
  
 public:
  ~Neuron();
  //methods
  //->
  void calc_output(int,float*);
  float somme_input(int,float*);
  void Init_weights(int);
  //<-
  void back_propagation_body(Neuron*,float*,int,int,int);
  void back_propagation_head(float*,float*,int,int);
  //get
  float get_output();
  float get_bias();
  float* get_weights();
  float get_error();
  //set
  void set_output(float);
  void set_modifiers(float (*)(float),float (*)(float));
  void set_bias(float);
  //print
  void print_weights(int);
};
#endif
