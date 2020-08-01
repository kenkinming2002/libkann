#include "ActivationFunction.hpp"

#include <cmath>

namespace activation
{
  namespace function
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

  namespace derivative
  {
    double tanh(double val)
    {
      auto tmp = std::cosh(val);
      return 1 / (tmp * tmp);
    }
  }
}
