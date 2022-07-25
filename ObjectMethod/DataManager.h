#ifndef DMANAGER
#define DMANAGER
#include <fstream>


#define DATA_SIZE 5000
#define F_L_SIZE 2
#define MAX_DATA_SET 10000
#define DIVIDER 100
#define TOP_T_DATA_SET 70
#define BOTTOM_T_DATA_SET 30

class DataManager
{
 private:
  float** training_input_data;
  float* training_output_data;
  int data_size, data_dimention;
 public:
  //constructor
  DataManager();
  //get
  void prepare_training_data();
  float** get_input_data();
  float* get_output_data();
  //gen
  void gen_training_data();
  float gen_output_data(float*);
  bool contains(float*, int);
  //save
  void save_training_data();
  //load
  void load_training_data();
  
};
#endif
