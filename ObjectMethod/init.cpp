#include "perceptron.h"
#include "NeuralNetwork.h"
#include "layer.h"
#include "Neuron.h"
#include <iostream>
#include <stdlib.h>
using namespace std;

int main(int argc, char** argv)
{
  std::cout << " +--------------------------------+"<<std::endl;
  std::cout << " |          Object METHOD         |"<<std::endl;
  std::cout << " |                                |"<<std::endl;
  std::cout << " +--------------------------------+"<<std::endl;
  std::cout << " made by K. LASTRA 06/2020"<<std::endl<<std::endl;

  int c = 1;
  
  NeuralNetwork NN;
  NN.training_data();
  NN.build();
  NN.load_neural_set();
  //NN.print_data_set();

  std::cout <<std::endl<< "NEURAL NETWORK MENU" << std::endl;
  std::cout << "Data settings:" << std::endl << std::endl;
  std::cout << std::endl;
  while(c)
  {
    std::cout << "Selection codes:" << std::endl;
    std::cout << "(1) learn" << std::endl;
    std::cout << "(2) test" << std::endl;
    std::cout << "(0) end" << std::endl;
    std::cout << "write selection code: ";
    std::cin >> c;
    std::cout << std::endl;
    if(c == 1)
    {
      int d = 0;
      int d_ = 0;
      std::cout << "set epochs number: ";
      std::cin >> d;
      std::cout << std::endl;
      std::cout << "set save epoch: ";
      std::cin >> d_;
      std::cout << std::endl;
      std::cout << "Start learning..." << std::endl;
      NN.learn(d, d_);
    }
    else if(c == 2)
    {
      float d = 0;
      float test[2];
      std::cout << "set test data:" << std::endl;
      std::cout << "x: ";
      std::cin >> d;
      test[0] = d;
      std::cout << std::endl;

      std::cout << "y: ";
      std::cin >> d;
      test[1] = d;
      std::cout << std::endl;
      NN.test(test);
      std::cout << std::endl;
    }
    else if(c != 0)
      std::cout << "error, incorrect code, try again" << std::endl; 
  }
  return 0;
}
