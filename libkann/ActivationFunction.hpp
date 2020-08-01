#pragma once

namespace activation
{
  namespace function
  {
    double sigmoid(double val);
    double tanh(double val);
  }

  namespace derivative
  {
    double tanh(double val);
  }
}
