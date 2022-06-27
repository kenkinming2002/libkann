#include <libkann/Math.hpp>

#include <Eigen/Eigen>

#include <range/v3/all.hpp>
#include <fmt/core.h>

namespace kann::math
{
  using EigenArray  = Eigen::ArrayXf;
  using EigenVector = Eigen::RowVectorXf;
  using EigenMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  static inline auto to_eigen_array(auto& tensor)
  {
    return EigenArray::Map(tensor.data(), tensor.size());
  }

  static inline auto to_eigen_vector(auto& tensor)
  {
    assert(tensor.rank() == 1);
    return EigenVector::Map(tensor.data(), tensor.shape().size());
  }

  static inline auto to_eigen_matrix(auto& tensor)
  {
    assert(tensor.rank() == 2);
    return EigenMatrix::Map(tensor.data(),
      tensor.shape().dimension(0),
      tensor.shape().dimension(1)
    );
  }

  float norm(const Tensor<float>& value)
  {
    Tensor<float> flattened = value.flatten();

    float sum = 0.0;
    for(size_t i=0; i<value.size(); ++i)
      sum += value(i) * value(i);

    return std::sqrt(sum);
  }

  void product(Tensor<float> dst,
      Tensor<const float> a, bool transpose_a,
      Tensor<const float> b, bool transpose_b)
  {
    assert(a.rank() == 2);
    assert(b.rank() == 2);
    assert(dst.rank() == 2);

    auto _a   = EigenMatrix::Map(a.data(),   a.shape().dimension(0),   a.shape().dimension(1));
    auto _b   = EigenMatrix::Map(b.data(),   b.shape().dimension(0),   b.shape().dimension(1));
    auto _dst = EigenMatrix::Map(dst.data(), dst.shape().dimension(0), dst.shape().dimension(1));
    if(transpose_a)
    {
      if(transpose_b)
        _dst.noalias() = _a.transpose() * _b.transpose();
      else
        _dst.noalias() = _a.transpose() * _b;
    }
    else
    {
      if(transpose_b)
        _dst.noalias() = _a * _b.transpose();
      else
        _dst.noalias() = _a * _b;
    }
  }

  inline Tensor<const float> pad(Tensor<const float> inputs, Vec2 padding_size)
  {
    if(padding_size.height() == 0 && padding_size.width() == 0)
      return inputs; // Fast path

    const auto [M, N] = std::make_pair(inputs.dimension(0), inputs.dimension(1));
    const Vec2 input_size = Vec2(inputs.dimension(2), inputs.dimension(3));
    const Vec2 output_size = input_size + 2 * padding_size;

    Tensor<float> outputs = Tensor<float>::create(Shape{M, N, output_size.height(), output_size.width()});
    for(size_t j=0; j<M; ++j)
      for(size_t i=0; i<N; ++i)
      {
        // Zero-fill
        for(size_t y=0; y<padding_size.height(); ++y)
          for(size_t x=0; x<output_size.width(); ++x)
            outputs(j, i, y, x) = 0.0f;

        for(size_t y=padding_size.height(); y<padding_size.height()+input_size.height(); ++y)
        {
          // Zero fill
          for(size_t x=0; x<padding_size.width(); ++x)
            outputs(j, i, y, x) = 0.0f;

          for(size_t x=padding_size.width(); x<padding_size.width()+input_size.width(); ++x)
            outputs(j, i, y, x) = inputs(j, i, y - padding_size.height(), x - padding_size.width());

          // Zero fill
          for(size_t x=padding_size.width() + input_size.width(); x<output_size.width(); ++x)
            outputs(j, i, y, x) = 0.0f;
        }

        // Zero-fill
        for(size_t y=padding_size.height()+input_size.height(); y<output_size.height(); ++y)
          for(size_t x=0; x<output_size.width(); ++x)
            outputs(j, i, y, x) = 0.0f;
      }

    return outputs;
  }

  void image2d_operation(Tensor<float> outputs,
      Tensor<const float> inputs, bool transpose_inputs,
      Tensor<const float> kernels, bool transpose_kernels,
      Image2DOperation operation)
  {
    auto [M1, N1, P1, P2] = std::make_tuple(outputs.dimension(0), outputs.dimension(1), outputs.dimension(2), outputs.dimension(3));
    auto [M2, K1, Q1, Q2] = std::make_tuple(inputs .dimension(0), inputs .dimension(1), inputs .dimension(2), inputs .dimension(3));
    auto [K2, N2, R1, R2] = std::make_tuple(kernels.dimension(0), kernels.dimension(1), kernels.dimension(2), kernels.dimension(3));

    if(transpose_inputs)  std::swap(M2, K1);
    if(transpose_kernels) std::swap(K2, N2);

    assert(M1 == M2);
    assert(N1 == N2);
    assert(K1 == K2);

    const auto [M, N, K] = std::make_tuple(M1, N1, K1);
    const auto [output_size, input_size, kernel_size] = std::make_tuple(Vec2(P1, P2), Vec2(Q1, Q2), Vec2(R1, R2));
    const Vec2 padding_size = ((output_size - input_size) + (kernel_size - Vec2(1,1))) / 2;
    inputs = pad(std::move(inputs), padding_size);

    // It would a mystery if the following code works the first time.
    // Nevertheless, we plan to replace it with algorithm using FFT anyway hopefully soonish
    outputs.fill(0.0);
    for(size_t j=0; j<M; ++j)
      for(size_t i=0; i<N; ++i)
        for(size_t k=0; k<K; ++k)
          for(size_t oy=0; oy<output_size.height(); ++oy)
            for(size_t ox=0; ox<output_size.width(); ++ox)
            {
              // We are inside five nested for loop and we can finally start doing the cross_correlation/convolution
              float sum = 0.0f;
              for(size_t ky = 0; ky<kernel_size.height(); ++ky)
                for(size_t kx = 0; kx<kernel_size.width(); ++kx)
                {
                  const auto [iy, ix] = std::make_pair(oy + ky, ox + kx); // This disregard padding

                  float input  = transpose_inputs  ? inputs(k, j, iy /*- padding_size.height()*/, ix /*- padding_size.width()*/)
                                                   : inputs(j, k, iy /*- padding_size.height()*/, ix /*- padding_size.width()*/);
                                                                                                   //
                  float kernel = (operation == Image2DOperation::CROSS_CORRELATION) ? (transpose_kernels ? kernels(i, k, ky,                            kx                          ) : kernels(k, i, ky,                            kx                          ))
                               : (operation == Image2DOperation::CONVOLUTION)       ? (transpose_kernels ? kernels(i, k, kernel_size.height() - ky - 1, kernel_size.width() - kx - 1) : kernels(k, i, kernel_size.height() - ky - 1, kernel_size.width() - kx - 1))
                               : (assert(false && "Unimplemented"), 0.0f);

                  sum += input * kernel;
                }

              outputs(j, i, oy, ox) += sum;
            }

  }
}
