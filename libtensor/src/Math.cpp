#include <libtensor/Math.hpp>

#include <libtensor/Backend.hpp>

#include <spdlog/spdlog.h>

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

  static inline sgecorr2d_t resolve_sgecorr2d()
  {
    for(const Backend& backend : backends())
      if(backend.sgecorr2d)
        return backend.sgecorr2d;

    spdlog::error("libtensor : No backend provide sgecorr2d function");
    exit(EXIT_FAILURE);
  }

  static inline void sgecorr2d(size_t M, size_t N, size_t K,
      const float* inputs,  size_t height_input, size_t width_input, bool trans_inputs,
      const float* kernels, size_t height_kernel, size_t width_kernel, bool trans_kernels,
      float* outputs, size_t height_output, size_t width_output)
  {
    static sgecorr2d_t impl = resolve_sgecorr2d();
    impl(M, N, K,
        inputs,  height_input,  width_input,  trans_inputs,
        kernels, height_kernel, width_kernel, trans_kernels,
        outputs, height_output, width_output);
  }

  static inline sgeconv2d_t resolve_sgeconv2d()
  {
    for(const Backend& backend : backends())
      if(backend.sgeconv2d)
        return backend.sgeconv2d;

    spdlog::error("libtensor : No backend provide sgeconv2d function");
    exit(EXIT_FAILURE);
  }

  static inline void sgeconv2d(size_t M, size_t N, size_t K,
      const float* inputs,  size_t height_input, size_t width_input, bool trans_inputs,
      const float* kernels, size_t height_kernel, size_t width_kernel, bool trans_kernels,
      float* outputs, size_t height_output, size_t width_output)
  {
    static sgeconv2d_t impl = resolve_sgeconv2d();
    impl(M, N, K,
        inputs,  height_input,  width_input,  trans_inputs,
        kernels, height_kernel, width_kernel, trans_kernels,
        outputs, height_output, width_output);
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

  void image2d_operation(Tensor<float> outputs,
      Tensor<const float> inputs, bool transpose_inputs,
      Tensor<const float> kernels, bool transpose_kernels,
      Image2DOperation operation)
  {
    size_t M1 = outputs.dimension(0), N1 = outputs.dimension(1), output_height = outputs.dimension(2), output_width = outputs.dimension(3);
    size_t M2 = inputs .dimension(0), K1 = inputs .dimension(1), input_height  = inputs .dimension(2), input_width  = inputs .dimension(3);
    size_t K2 = kernels.dimension(0), N2 = kernels.dimension(1), kernel_height = kernels.dimension(2), kernel_width = kernels.dimension(3);

    if(transpose_inputs)  std::swap(M2, K1);
    if(transpose_kernels) std::swap(K2, N2);

    assert(M1 == M2);
    assert(N1 == N2);
    assert(K1 == K2);

    const auto [M, N, K] = std::make_tuple(M1, N1, K1);
    switch(operation)
    {
    case Image2DOperation::CROSS_CORRELATION:
      sgecorr2d(M, N, K,
          inputs.data(),  input_height,  input_width,  transpose_inputs,
          kernels.data(), kernel_height, kernel_width, transpose_kernels,
          outputs.data(), output_height, output_width);
      break;
    case Image2DOperation::CONVOLUTION:
      sgeconv2d(M, N, K,
          inputs.data(),  input_height,  input_width,  transpose_inputs,
          kernels.data(), kernel_height, kernel_width, transpose_kernels,
          outputs.data(), output_height, output_width);
      break;
    }
  }
}
