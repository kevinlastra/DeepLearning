#include "mathF.h"
#include "NeuralNetwork.h"
#include "DataManager.h"
#include <iostream>
#include <unistd.h>
#include <fstream>
#include <string>
#include <fstream>

void NeuralNetwork::build()
{
  std::cout << "Starting Neural network"<<std::endl;
  std::cout << "Testing data range ("<<0<<" ,"<<DIVIDER<<")"<<std::endl;
  std::cout << "Square area -> Bottom: "<<BOTTOM_T_DATA_SET<<"  Top: "<<TOP_T_DATA_SET<<std::endl;
  layer* layers = new layer[size];
  f_layer = &layers[0];
  l_layer = &layers[size-1];
  f_layer->set_connections(NULL,&layers[1]);
  l_layer->set_connections(&layers[size-2],NULL);
  for(int i = 0;i < size;i++)
  {
    if(i > 0 && i < size -1)
      layers[i].set_connections(&layers[i-1],&layers[i+1]);
    layers[i].gen_layer(i,structure[i],sigmoid,sigmoid_d);
  }
}
NeuralNetwork::~NeuralNetwork()
{
  delete[] training_outputs;
  for(int i = 0;i < DATA_SIZE;i++)
  {
    delete[] training_inputs[i];
  }
  delete[] training_inputs;
}
void NeuralNetwork::training_data()
{
  DataManager dm;
  dm.prepare_training_data();
  training_inputs = dm.get_input_data();
  training_outputs = dm.get_output_data();
  Graph_data_training(training_inputs, training_outputs, "Training_Graph.ps");
}
void NeuralNetwork::learn(int epochs, int save_epoch)
{
  float** data = new float*[DATA_SIZE];
  int saves_n = 1;
  float test_[2];
  for(int j = 0; j < epochs;j++)
  {
    for(int i = 0;i < DATA_SIZE;i++)
    {
      f_layer->set_output(training_inputs[i]);
      f_layer->next_layer->calc_outputs();
      l_layer->back_propagation(i,f_layer,training_outputs);
      if(j%(save_epoch) == 0 && j != 0)
	data[i] = l_layer->get_output();
    }
    if(j%(save_epoch) == 0 && j != 0)
    {
      std::cout <<saves_n<<".saving weights"  << std::endl;
      saves_n++;
      save_neural_set();
      //print_output(data);
      Graph_data_output(training_inputs, data, "Output_Graph.ps");
    }
  }
  std::cout << "Learn end." << std::endl;
  save_neural_set();
  Graph_data_output(training_inputs, data, "Output_Graph.ps");
  for(int i = 0; i < DATA_SIZE;i++)
    delete[] data[i];
  delete[] data;
}
void NeuralNetwork::test(float* data)
{
  DataManager dm;
  int k = dm.gen_output_data(data);
  for(int i = 0; i < F_L_SIZE;i++)
  {
    std::cout << "data["<<i<<"]: "<<data[i]<<std::endl;
    data[i] = normalize(data[i],100);
    std::cout << "data_normalized["<<i<<"]: "<<data[i]<<std::endl;
  }
  f_layer->set_output(data);
  f_layer->next_layer->calc_outputs();
    std::cout << "TEST OUTPUT = " << l_layer->get_output()[0] <<"      DESIRED OUTPUT = "<<k<< std::endl; 
}
void NeuralNetwork::save_neural_set()
{
  layer* alayer = f_layer->next_layer;
  float* weights;

  std::ofstream file;
  file.open("Data/neural_set.txt");
  while(alayer != NULL)
  {
    for(int i = 0; i < alayer->get_size();i++)
    {
      weights = alayer->get_neuron(i)->get_weights();
      file << alayer->prev_layer->get_size() << "_";
      for(int j = 0;j < alayer->prev_layer->get_size();j++)
      {
	file << weights[j]<<"_";
      }
      file << alayer->get_neuron(i)->get_bias() << "\n";
    }
    alayer = alayer->next_layer;
  }
  file.close();
}
void NeuralNetwork::load_neural_set()
{
  std::ifstream file("Data/neural_set.txt");
  std::string line;
  int npos = 0;
  int prev_pos = 0;
  int size_ = 0;
  layer* alayer = f_layer->next_layer;
  int layer_size;
  int neuron_index = 0;
  float data = 0;
  if(file.is_open())
  {
    while(getline(file,line))
    {
      layer_size = alayer->get_size();
      npos = line.find_first_of("_",prev_pos);
      size_ = stoi(line.substr(prev_pos,npos));
      prev_pos = npos;
      for(int i = 0;i < size_;i++)
      {
	npos = line.find_first_of("_",prev_pos+1);
	
	alayer->get_neuron(neuron_index)->get_weights()[i] = stof(line.substr(prev_pos+1,npos));
	prev_pos = npos;
      }
      npos = line.find_first_of("_",prev_pos+1);
      alayer->get_neuron(neuron_index)->set_bias(stof(line.substr(prev_pos+1,npos)));
      prev_pos = 0;
      npos = 0;
      neuron_index++;
      if(neuron_index == layer_size)
      {
	alayer = alayer->next_layer;
	neuron_index = 0;
      }
      if(alayer == NULL)
	break;
    }
    std::cout << "Data set loaded" << std::endl;
  }
  else
    std::cout << "error neural data doesn't found" << std::endl;
  file.close();
}
void NeuralNetwork::print_output(float** output)
{
  //std::cout << "\033[2J"<<std::endl;
  std::cout << std::endl<<std::endl<<"OUTPUT:" << std::endl;
  for(int i = 0;i < DATA_SIZE;i++)
  {
    std::cout <<"data["<<i<<"] -->   x: "<<training_inputs[i][0]<< "      y: "<<
      training_inputs[i][1]<<"    calc_output: "<<output[i][0] << "     wanted_output: " <<training_outputs[i]<<std::endl;
  }
  usleep(10);
}
void NeuralNetwork::print_data_set()
{
  std::cout << "INPUT                 OUTPUT" << std::endl;
  for(int i = 0; i< DATA_SIZE;i++)
  {
    for(int j = 0; j < F_L_SIZE;j++)
    {
      std::cout <<training_inputs[i][j]<<"   ";
    }
    std::cout <<"             "<<training_outputs[i]<<std::endl;
  }
}
void NeuralNetwork::Graph_data_training(float** t_input, float* t_output, std::string doc_name)
{
  float x,y;
  int cpt=0;
  std::ofstream output;                           
  output.open(doc_name,std::ios::out);
  output << "%!PS-Adobe-3.0" << std::endl;
  output << "%%BoundingBox: 0 0 612 792" << std::endl;
  output << std::endl;
  for(int i=0;i<DATA_SIZE;i++)
  {
    x = t_input[i][0]*612;
    y = t_input[i][1]*792;
    output << x << " " << y << " 1 0 360 arc" <<std::endl;
    if(t_output[i] == 1)
    {
      output << "0 1 0 setrgbcolor" <<std::endl;
    }
    else
    {
      output << "1 0 0 setrgbcolor" <<std::endl;
    }
    output << "fill" <<std::endl;
    output << "stroke"<<std::endl;
    output << std::endl;
  }
  output << std::endl;
  output << "showpage";
  output << std::endl;
  output.close();
}
void NeuralNetwork::Graph_data_output(float** t_input, float** t_output, std::string doc_name)
{
  float x,y;
  std::ofstream output;                           
  output.open(doc_name,std::ios::out);
  output << "%!PS-Adobe-3.0" << std::endl;
  output << "%%BoundingBox: 0 0 612 792" << std::endl;
  output << std::endl;
  for(int i=0;i<DATA_SIZE;i++)
  {
    x = t_input[i][0]*612;
    y = t_input[i][1]*792;
    output << x << " " << y << " 1 0 360 arc" <<std::endl;
    if(t_output[i][0] > 0.5)
    {
      output << "0 1 0 setrgbcolor" <<std::endl;
    }
    else
    {
      output << "1 0 0 setrgbcolor" <<std::endl;
    }
    output << "fill" <<std::endl;
    output << "stroke"<<std::endl;
    output << std::endl;
  }
  output << std::endl;
  output << "showpage";
  output << std::endl;
  output.close();
}
