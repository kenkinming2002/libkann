#pragma once

#include <libkann/export.hpp>

struct ActivationFunction
{
public:
  typedef double (*function_t)(double);

public:
  constexpr ActivationFunction(function_t normal, function_t derivative) 
    : normal(normal), derivative(derivative) {}

public:
  function_t normal;
  function_t derivative;
};

namespace activation_function
{
  LIBKANN_SYMEXPORT extern ActivationFunction sigmoid;
  LIBKANN_SYMEXPORT extern ActivationFunction tanh;
}
