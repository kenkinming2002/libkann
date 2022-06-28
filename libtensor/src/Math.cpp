#include <libtensor/Math.hpp>

#include <libtensor/Backend.hpp>

#include <spdlog/spdlog.h>
#include <range/v3/all.hpp>
#include <fmt/core.h>

namespace tensor::math
{
  float norm(const Tensor<float>& value)
  {
    Tensor<float> flattened = value.flatten();

    float sum = 0.0;
    for(size_t i=0; i<value.size(); ++i)
      sum += value(i) * value(i);

    return std::sqrt(sum);
  }

  static inline sgemm_t resolve_sgemm()
  {
    for(const Backend& backend : backends())
      if(backend.sgemm)
        return backend.sgemm;

    spdlog::error("libtensor : No backend provide sgemm function");
    exit(EXIT_FAILURE);
  }

  static inline void sgemm(size_t M, size_t N, size_t K, const float* A, bool trans_a, const float* B, bool trans_b, float* C)
  {
    static sgemm_t impl = resolve_sgemm();
    return impl(M, N, K, A, trans_a, B, trans_b, C);
  }

  void product(Tensor<float> dst,
      Tensor<const float> a, bool transpose_a,
      Tensor<const float> b, bool transpose_b)
  {
    assert(a.rank() == 2);
    assert(b.rank() == 2);
    assert(dst.rank() == 2);

    const size_t M1 = dst.dimension(0);
    const size_t N1 = dst.dimension(1);

    size_t M2 = a.dimension(0), K1 = a.dimension(1);
    size_t K2 = b.dimension(0), N2 = b.dimension(1);

    if(transpose_a) std::swap(M2, K1);
    if(transpose_b) std::swap(K2, N2);

    assert(M1 == M2);
    assert(N1 == N2);
    assert(K1 == K2);

    const size_t M = M1, N = N1, K = K1;
    sgemm(M, N, K, a.data(), transpose_a, b.data(), transpose_b, dst.data());
  }

  inline Tensor<const float> pad(Tensor<const float> inputs, size_t padding_height, size_t padding_width)
  {
    if(padding_height == 0 && padding_width == 0)
      return inputs; // Fast path

    const auto [M, N] = std::make_pair(inputs.dimension(0), inputs.dimension(1));
    const auto [input_height, input_width]   = std::make_pair(inputs.dimension(2), inputs.dimension(3));
    const auto [output_height, output_width] = std::make_pair(input_height + 2 * padding_height,
                                                              input_width +  2 * padding_width);

    Tensor<float> outputs = Tensor<float>::create(Shape{M, N, output_height, output_width});
    for(size_t j=0; j<M; ++j)
      for(size_t i=0; i<N; ++i)
      {
        // Zero-fill
        for(size_t y=0; y<padding_height; ++y)
          for(size_t x=0; x<output_width; ++x)
            outputs(j, i, y, x) = 0.0f;

        for(size_t y=padding_height; y<padding_height+input_height; ++y)
        {
          // Zero fill
          for(size_t x=0; x<padding_width; ++x)
            outputs(j, i, y, x) = 0.0f;

          for(size_t x=padding_width; x<padding_width+input_width; ++x)
            outputs(j, i, y, x) = inputs(j, i, y - padding_height, x - padding_width);

          // Zero fill
          for(size_t x=padding_width + input_width; x<output_width; ++x)
            outputs(j, i, y, x) = 0.0f;
        }

        // Zero-fill
        for(size_t y=padding_height+input_height; y<output_height; ++y)
          for(size_t x=0; x<output_width; ++x)
            outputs(j, i, y, x) = 0.0f;
      }

    return outputs;
  }

  void image2d_operation(Tensor<float> outputs,
      Tensor<const float> inputs, bool transpose_inputs,
      Tensor<const float> kernels, bool transpose_kernels,
      Image2DOperation operation)
  {
    auto [M1, N1, output_height, output_width] = std::make_tuple(outputs.dimension(0), outputs.dimension(1), outputs.dimension(2), outputs.dimension(3));
    auto [M2, K1, input_height,  input_width ] = std::make_tuple(inputs .dimension(0), inputs .dimension(1), inputs .dimension(2), inputs .dimension(3));
    auto [K2, N2, kernel_height, kernel_width] = std::make_tuple(kernels.dimension(0), kernels.dimension(1), kernels.dimension(2), kernels.dimension(3));

    if(transpose_inputs)  std::swap(M2, K1);
    if(transpose_kernels) std::swap(K2, N2);

    assert(M1 == M2);
    assert(N1 == N2);
    assert(K1 == K2);

    const auto [M, N, K] = std::make_tuple(M1, N1, K1);
    //const auto [output_size, input_size, kernel_size] = std::make_tuple(Vec2(P1, P2), Vec2(Q1, Q2), Vec2(R1, R2));

    const auto [padding_height, padding_width] = std::make_pair((output_height - input_height + kernel_height - 1) / 2,
                                                                (output_width  - input_width  + kernel_width  - 1) / 2);

    inputs = pad(std::move(inputs), padding_height, padding_width);

    // It would a mystery if the following code works the first time.
    // Nevertheless, we plan to replace it with algorithm using FFT anyway hopefully soonish
    outputs.fill(0.0);
    for(size_t j=0; j<M; ++j)
      for(size_t i=0; i<N; ++i)
        for(size_t k=0; k<K; ++k)
          for(size_t oy=0; oy<output_height; ++oy)
            for(size_t ox=0; ox<output_width; ++ox)
            {
              // We are inside five nested for loop and we can finally start doing the cross_correlation/convolution
              float sum = 0.0f;
              for(size_t ky = 0; ky<kernel_height; ++ky)
                for(size_t kx = 0; kx<kernel_width; ++kx)
                {
                  const auto [iy, ix] = std::make_pair(oy + ky, ox + kx); // This disregard padding

                  float input  = transpose_inputs  ? inputs(k, j, iy /*- padding_height*/, ix /*- padding_width*/)
                                                   : inputs(j, k, iy /*- padding_height*/, ix /*- padding_width*/);
                                                                                                   //
                  float kernel = (operation == Image2DOperation::CROSS_CORRELATION) ? (transpose_kernels ? kernels(i, k, ky,                     kx                   ) : kernels(k, i, ky,                     kx                   ))
                               : (operation == Image2DOperation::CONVOLUTION)       ? (transpose_kernels ? kernels(i, k, kernel_height - ky - 1, kernel_width - kx - 1) : kernels(k, i, kernel_height - ky - 1, kernel_width - kx - 1))
                               : (assert(false && "Unimplemented"), 0.0f);

                  sum += input * kernel;
                }

              outputs(j, i, oy, ox) += sum;
            }

  }
}
