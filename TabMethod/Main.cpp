
#include "NeuralNetwork.h"
#include <iostream>
int main()
{
  std::cout << " +--------------------------------+"<<std::endl;
  std::cout << " |          TAB METHOD            |"<<std::endl;
  std::cout << " |                                |"<<std::endl;
  std::cout << " +--------------------------------+"<<std::endl;
  std::cout << " | learn if a coord is in or out  |"<<std::endl;
  std::cout << " | an area, which in this case is |"<<std::endl;
  std::cout << " | superior than 2 and inferior   |"<<std::endl;
  std::cout << " | than 9                         |"<<std::endl;
  std::cout << " +--------------------------------+"<<std::endl;

  std::cout << " made by K. LASTRA 06/2020"<<std::endl<<std::endl;

  int sl[] = {2,3,1};
  std::cout << " BUILDING NEURAL NETWORK"<<std::endl;
  std::cout << " Input: "<<2<<std::endl;
  std::cout << " Nb hidden layers: "<<1<<std::endl;
  std::cout << " Length of hidden layer: "<<3<<std::endl;
  std::cout << " Output: "<<1<<std::endl;
  std::cout << " Size of data test: "<<20<<std::endl<<std::endl;
  std::cout << " Nb learn cycles: "<<1000000<<std::endl;

  NeuralNetwork nn(20,3,sl);
  nn.print_IO();
  char a;

  std::cout << "Print neural network graph? (y/n)" << std::endl;
  std::cin >> a;
  if(a == 'y' || a == 'Y')
    nn.dot_data();
  
  std::cout << "Start learning? (y/n)" << std::endl;
  std::cin >> a;
  if(a != 'y' && a != 'Y')
    return 0;
  
  std::cout << "  0%";
  for(int i = 0;i < 1000000;i++)
  {
    if(i%10000 == 0)
    {
      printf("\b\b\b\b%3d\%",i/10000);
      fflush(stdout);
    }
    nn.learn();
  }
  std::cout << std::endl;
  std::cout << "Open neural network graph after learned? (y/n)" << std::endl;
  std::cin >> a;
  if(a == 'y' || a == 'Y')
    nn.dot_data();
  
  while(true)
  {
    std::cout << "Test? (y/n)" << std::endl;
    std::cin >> a;
    if(a != 'y' && a != 'Y')
      break;
    nn.test_data();
  }
  return 0;
}
