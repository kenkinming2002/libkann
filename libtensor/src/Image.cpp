#include <libtensor/Image.hpp>

namespace tensor
{
  template Tensor<float>       image2d_cross_correlate(Tensor<const float>       inputs, bool trans_inputs, Tensor<const float>       kernels, bool trans_kernels, Vec2 output_size);
  template Tensor<double>      image2d_cross_correlate(Tensor<const double>      inputs, bool trans_inputs, Tensor<const double>      kernels, bool trans_kernels, Vec2 output_size);
  template Tensor<long double> image2d_cross_correlate(Tensor<const long double> inputs, bool trans_inputs, Tensor<const long double> kernels, bool trans_kernels, Vec2 output_size);

  template Tensor<float>       image2d_convolve(Tensor<const float>       inputs, bool trans_inputs, Tensor<const float>       kernels, bool trans_kernels, Vec2 output_size);
  template Tensor<double>      image2d_convolve(Tensor<const double>      inputs, bool trans_inputs, Tensor<const double>      kernels, bool trans_kernels, Vec2 output_size);
  template Tensor<long double> image2d_convolve(Tensor<const long double> inputs, bool trans_inputs, Tensor<const long double> kernels, bool trans_kernels, Vec2 output_size);
}
