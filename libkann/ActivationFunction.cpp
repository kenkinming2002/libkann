#include "ActivationFunction.hpp"

#include <cmath>

namespace
{
  double sigmoid(double val)
  { 
    return 1.0 /  (1.0 + std::exp(-val)); 
  }

  double sigmoidDerivative(double val)
  { 
    auto tmp = std::exp(-val);
    return tmp / ((1+tmp) * (1+tmp));
  }

  double tanh(double val)
  {
    return std::tanh(val);
  }

  double tanhDerivative(double val)
  {
    auto tmp = std::cosh(val);
    return 1 / (tmp * tmp);
  }
}

namespace activation_function
{
  ActivationFunction sigmoid(::sigmoid, ::sigmoidDerivative);
  ActivationFunction tanh(::tanh, ::tanhDerivative);
}
