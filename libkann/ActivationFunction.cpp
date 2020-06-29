#include "ActivationFunction.hpp"

#include <cmath>

namespace activation_function
{
  double sigmoid(double val)
  { 
    return 1.0 /  (1.0 + std::exp(-val)); 
  }

  double tanh(double val)
  {
    return std::tanh(val);
  }
}
