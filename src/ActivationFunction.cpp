#include <libkann/ActivationFunction.hpp>

#include <cmath>
#include <stdexcept>

namespace kann
{
  double ActivationFunction::normal(double val) const
  {
    auto identity = [](double val) {
      return val;
    };
    auto sigmoid = [](double val) {
      return 1.0 /  (1.0 + std::exp(-val));
    };
    auto tanh = [](double val) {
      return std::tanh(val);
    };

    switch(type)
    {
      case Type::IDENTITY:
        return identity(val);
      case Type::SIGMOID:
        return sigmoid(val);
      case Type::TANH:
        return tanh(val);
      default:
        throw std::runtime_error("Invalid activation function type");
    }
  }

  double ActivationFunction::derivative(double val) const
  {
    auto identityDerivative = [](double /*val*/) {
      return 1.0;
    };
    auto sigmoidDerivative = [](double val) {
      auto tmp = std::exp(-val);
      return tmp / ((1+tmp) * (1+tmp));
    };
    auto tanhDerivative = [](double val) {
      auto tmp = std::cosh(val);
      return 1 / (tmp * tmp);
    };

    switch(type)
    {
      case Type::IDENTITY:
        return identityDerivative(val);
      case Type::SIGMOID:
        return sigmoidDerivative(val);
      case Type::TANH:
        return tanhDerivative(val);
      default:
        throw std::runtime_error("Invalid activation function type");
    }
  }
}

