#include <libtensor/Image.hpp>

#include <libtensor/details/Eigen.hpp>

namespace tensor
{
  template<typename T>
  Tensor<const T> image2d_cross_correlate(Tensor<const T> inputs, bool trans_inputs, Tensor<const T> kernels, bool trans_kernels, Vec2 output_size)
  {
    auto M = inputs .dimension(0), K1 = inputs .dimension(1);
    auto K2 = kernels.dimension(0), N = kernels.dimension(1);

    if(trans_inputs)  std::swap(M, K1);
    if(trans_kernels) std::swap(K2, N);

    assert(K1 == K2);
    auto K = K1;

    auto input_size  = Vec2(inputs.dimension(2), inputs.dimension(3));
    auto kernel_size = Vec2(kernels.dimension(2), kernels.dimension(3));
    auto padding_size = (kernel_size + output_size - input_size - Vec2(1,1)) / 2;

    auto outputs = Tensor<T>::create(Shape::make(M, N, output_size.height(), output_size.width()));

    auto _inputs  = details::to_tensor4d(inputs,  trans_inputs);
    auto _kernels = details::to_tensor4d(kernels, trans_kernels);
    auto _outputs = details::to_tensor4d(outputs, false);

    _outputs.setZero();
    for(size_t m=0; m<M; ++m)
      for(size_t n=0; n<N; ++n)
        for(size_t k=0; k<K; ++k)
        {
          auto _input  = _inputs .chip(m, 0).chip(k, 0);
          auto _kernel = _kernels.chip(k, 0).chip(n, 0);
          auto _output = _outputs.chip(m, 0).chip(n, 0);

          Eigen::array<std::pair<int, int>, 2> paddings;
          paddings[0] = std::make_pair(padding_size.height(), padding_size.height());
          paddings[1] = std::make_pair(padding_size.width(),  padding_size.width());
          auto _padded_input = _input.pad(paddings);

          const Eigen::array<ptrdiff_t, 2> dims({0, 1});
          _output += _padded_input.convolve(_kernel, dims);
        }

    return outputs;
  }

  template<typename T>
  Tensor<const T> image2d_convolve(Tensor<const T> inputs, bool trans_inputs, Tensor<const T> kernels, bool trans_kernels, Vec2 output_size)
  {
    auto M = inputs .dimension(0), K1 = inputs .dimension(1);
    auto K2 = kernels.dimension(0), N = kernels.dimension(1);

    if(trans_inputs)  std::swap(M, K1);
    if(trans_kernels) std::swap(K2, N);

    assert(K1 == K2);
    auto K = K1;

    auto input_size  = Vec2(inputs.dimension(2), inputs.dimension(3));
    auto kernel_size = Vec2(kernels.dimension(2), kernels.dimension(3));
    auto padding_size = (kernel_size + output_size - input_size - Vec2(1,1)) / 2;

    auto outputs = Tensor<T>::create(Shape::make(M, N, output_size.height(), output_size.width()));

    auto _inputs  = details::to_tensor4d(inputs,  trans_inputs);
    auto _kernels = details::to_tensor4d(kernels, trans_kernels);
    auto _outputs = details::to_tensor4d(outputs, false);

    _outputs.setZero();
    for(size_t m=0; m<M; ++m)
      for(size_t n=0; n<N; ++n)
        for(size_t k=0; k<K; ++k)
        {
          auto _input  = _inputs .chip(m, 0).chip(k, 0);
          auto _kernel = _kernels.chip(k, 0).chip(n, 0);
          auto _output = _outputs.chip(m, 0).chip(n, 0);

          Eigen::array<std::pair<int, int>, 2> paddings;
          paddings[0] = std::make_pair(padding_size.height(), padding_size.height());
          paddings[1] = std::make_pair(padding_size.width(),  padding_size.width());
          auto _padded_input = _input.pad(paddings);

          const Eigen::array<ptrdiff_t, 2> dims({0, 1});
          const Eigen::array<bool, 2> reverse({true, true});
          _output += _padded_input.convolve(_kernel.reverse(reverse), dims);
        }

    return outputs;
  }

  template Tensor<const float>       image2d_cross_correlate(Tensor<const float>       inputs, bool trans_inputs, Tensor<const float>       kernels, bool trans_kernels, Vec2 output_size);
  template Tensor<const double>      image2d_cross_correlate(Tensor<const double>      inputs, bool trans_inputs, Tensor<const double>      kernels, bool trans_kernels, Vec2 output_size);
  template Tensor<const long double> image2d_cross_correlate(Tensor<const long double> inputs, bool trans_inputs, Tensor<const long double> kernels, bool trans_kernels, Vec2 output_size);

  template Tensor<const float>       image2d_convolve(Tensor<const float>       inputs, bool trans_inputs, Tensor<const float>       kernels, bool trans_kernels, Vec2 output_size);
  template Tensor<const double>      image2d_convolve(Tensor<const double>      inputs, bool trans_inputs, Tensor<const double>      kernels, bool trans_kernels, Vec2 output_size);
  template Tensor<const long double> image2d_convolve(Tensor<const long double> inputs, bool trans_inputs, Tensor<const long double> kernels, bool trans_kernels, Vec2 output_size);
}
