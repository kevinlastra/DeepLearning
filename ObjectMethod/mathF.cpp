#include "mathF.h"
#include <cmath>
#include <iostream>
float sigmoid(float x)
{
  return 1/(1+std::exp(-x));
}
float sigmoid_d(float x)
{
  return x*(1-x);
}
float normalize(float input, float divider)//[0, 1]
{
  return (input/divider);
}
