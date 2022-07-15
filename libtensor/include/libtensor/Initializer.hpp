#pragma once

#include <libtensor/Export.hpp>
#include <libtensor/Tensor.hpp>

namespace tensor
{
  template<typename T> LIBTENSOR_EXPORT Tensor<const T> create_constant(tensor::Shape shape, T value);
  template<typename T> LIBTENSOR_EXPORT Tensor<const T> create_normal(tensor::Shape shape, T mean, T stddev, std::default_random_engine& prng);
  template<typename T> LIBTENSOR_EXPORT Tensor<const T> create_uniform(tensor::Shape shape, T a, T b, std::default_random_engine& prng);
}


