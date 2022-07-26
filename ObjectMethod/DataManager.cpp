#include <iostream>
#include <string>
#include "DataManager.h"
#include "mathF.h"
DataManager::DataManager()
{
  training_input_data = new float*[DATA_SIZE];
  training_output_data = new float[DATA_SIZE];
  for(int i = 0;i < DATA_SIZE;i++)
  {
    training_input_data[i] = new float[F_L_SIZE];
  }
}
void DataManager::prepare_training_data()
{
  std::ifstream file("Data/neural_data_set.txt");
  if(file.is_open())
  {
    file.close();
    load_training_data();
    std::cout << "Training data loaded" << std::endl;
  }
  else
  {
    std::cout << "Training data not loaded." << std::endl;
    gen_training_data();
    save_training_data();
    std::cout << "Training data generated, trainig data saved." << std::endl;
  }
}
float** DataManager::get_input_data(){return training_input_data;}
float* DataManager::get_output_data(){return training_output_data;}
//gen
void DataManager::gen_training_data()
{
  int i = 0;
  int t_data = DATA_SIZE/2;
  int f_data = DATA_SIZE/2;
  srand(time(NULL));
  while(i < DATA_SIZE)
  {
    for(int j = 0; j < F_L_SIZE; j++)
    {
      training_input_data[i][j] = (float)(rand()%MAX_DATA_SET)/DIVIDER;
    }
    if(!contains(training_input_data[i],i))
    {
      training_output_data[i] = gen_output_data(training_input_data[i]);

      if((training_output_data[i] == 1 && t_data > 0) ||
	 (training_output_data[i] == 0 && f_data > 0))
      {
	if(training_output_data[i] == 1)
	  t_data--;
	else
	  f_data--;
	i++;
      }
    }
  }
  for(int i = 0; i < DATA_SIZE; i++)
  {
    for(int j = 0; j < F_L_SIZE; j++)
    {
      training_input_data[i][j] = normalize(training_input_data[i][j], 100);
    }
  }
}
float DataManager::gen_output_data(float* data)
{
  float test = 1;
  for(int i = 0; i < F_L_SIZE;i++)
    test = (test && data[i] >= BOTTOM_T_DATA_SET && data[i] <= TOP_T_DATA_SET)?1:0;
  return test;
}
bool DataManager::contains(float* data, int top)
{
  int i = 0;
  while(i < top && (data[0] != training_input_data[i][0]
		    || data[1] != training_input_data[i][1])){i++;}
  return !(i == top);
}
//save
void DataManager::save_training_data()
{
  std::ofstream file;
  file.open("Data/neural_data_set.txt");
  for(int i = 0;i < DATA_SIZE;i++)
  {
    file << training_input_data[i][0] << "_";
    file << training_input_data[i][1] << "_";
    file << training_output_data[i] << "\n";
  }
  file.close();
}
//load
void DataManager::load_training_data()
{
  std::ifstream file("Data/neural_data_set.txt");
  std::string line;
  int c_pos, prev_c_pos = 0;
  int i = 0;
  while(getline(file,line))
  {
    c_pos = line.find_first_of("_",prev_c_pos);
    training_input_data[i][0] = stof(line.substr(prev_c_pos,c_pos));
    prev_c_pos = c_pos+1;
    c_pos = line.find_first_of("_",prev_c_pos);
    training_input_data[i][1] = stof(line.substr(prev_c_pos,c_pos));
    
    training_output_data[i] = stof(line.substr(c_pos+1));
    prev_c_pos = 0;
    i++;
  }
  file.close();
}
