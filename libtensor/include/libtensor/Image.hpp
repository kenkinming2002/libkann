#pragma once

// Declaration
#include <libtensor/Export.hpp>
#include <libtensor/Tensor.hpp>
#include <libtensor/Vec.hpp>

// There is something common to most operation,
// we first extract the semantic information on shape of the inputs
// Create the outputs
// and run the operation, by most probably calling to an external library.
// The trouble lies in having to support multidimensional data, always

namespace tensor
{
  template<typename T>
  LIBTENSOR_EXPORT Tensor<const T> image2d_cross_correlate(Tensor<const T> inputs, bool trans_inputs, Tensor<const T> kernels, bool trans_kernels, Vec2 output_size);

  template<typename T>
  LIBTENSOR_EXPORT Tensor<const T> image2d_convolve(Tensor<const T> inputs, bool trans_inputs, Tensor<const T> kernels, bool trans_kernels, Vec2 output_size);
}
