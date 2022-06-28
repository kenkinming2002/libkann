#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>

extern "C"
{
  using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  void sgemm(size_t M, size_t N, size_t K,
      const float* A, bool trans_a,
      const float* B, bool trans_b,
      float* C)
  {
    auto a = trans_a ? Matrix::Map(A, K, M) : Matrix::Map(A, M, K);
    auto b = trans_b ? Matrix::Map(B, N, K) : Matrix::Map(B, K, N);
    auto c = Matrix::Map(C, M, N);
    if(trans_a)
    {
      if(trans_b)
        c = a.transpose() * b.transpose();
      else
        c = a.transpose() * b;
    }
    else
    {
      if(trans_b)
        c = a * b.transpose();
      else
        c = a * b;
    }
  }

  void sgecorr2d(size_t M, size_t N, size_t K,
      const float* inputs,  size_t height_input, size_t width_input, bool trans_inputs,
      const float* kernels, size_t height_kernel, size_t width_kernel, bool trans_kernels,
      float* outputs, size_t height_output, size_t width_output)
  {
    for(size_t m=0; m<M; ++m)
      for(size_t n=0; n<N; ++n)
        for(size_t k=0; k<K; ++k)
        {
          const size_t input_index  = trans_inputs  ? (k * M + m) : (m * K + k);
          const size_t kernel_index = trans_kernels ? (n * K + k) : (k * N + n);
          const size_t output_index = m * N + n;

          Eigen::TensorMap<const Eigen::Tensor<float, 2>> input (&inputs [input_index  * height_input  * width_input],  height_input,  width_input);
          Eigen::TensorMap<const Eigen::Tensor<float, 2>> kernel(&kernels[kernel_index * height_kernel * width_kernel], height_kernel, width_kernel);
          Eigen::TensorMap<      Eigen::Tensor<float, 2>> output(&outputs[output_index * height_output * width_output], height_output, width_output);

          const size_t height_padding = (height_output - height_input + height_kernel - 1) / 2;
          const size_t width_padding  = (width_output  - width_input  + width_kernel  - 1) / 2;

          Eigen::array<std::pair<int, int>, 2> paddings;
          paddings[0] = std::make_pair(height_padding, height_padding);
          paddings[1] = std::make_pair(width_padding,  width_padding);
          auto padded_input = input.pad(paddings);

          const Eigen::array<ptrdiff_t, 2> dims({0, 1});
          output = padded_input.convolve(kernel.reverse(dims), dims);
        }
  }

  void sgeconv2d(size_t M, size_t N, size_t K,
      const float* inputs,  size_t height_input, size_t width_input, bool trans_inputs,
      const float* kernels, size_t height_kernel, size_t width_kernel, bool trans_kernels,
      float* outputs, size_t height_output, size_t width_output)
  {
    for(size_t m=0; m<M; ++m)
      for(size_t n=0; n<N; ++n)
        for(size_t k=0; k<K; ++k)
        {
          const size_t input_index  = trans_inputs  ? (k * M + m) : (m * K + k);
          const size_t kernel_index = trans_kernels ? (n * K + k) : (k * N + n);
          const size_t output_index = m * N + n;

          Eigen::TensorMap<const Eigen::Tensor<float, 2>> input (&inputs [input_index  * height_input  * width_input],  height_input,  width_input);
          Eigen::TensorMap<const Eigen::Tensor<float, 2>> kernel(&kernels[kernel_index * height_kernel * width_kernel], height_kernel, width_kernel);
          Eigen::TensorMap<      Eigen::Tensor<float, 2>> output(&outputs[output_index * height_output * width_output], height_output, width_output);

          const size_t height_padding = (height_output - height_input + height_kernel - 1) / 2;
          const size_t width_padding  = (width_output  - width_input  + width_kernel  - 1) / 2;

          Eigen::array<std::pair<int, int>, 2> paddings;
          paddings[0] = std::make_pair(height_padding, height_padding);
          paddings[1] = std::make_pair(width_padding,  width_padding);
          auto padded_input = input.pad(paddings);

          const Eigen::array<ptrdiff_t, 2> dims({0, 1});
          output = padded_input.convolve(kernel, dims);
        }
  }
}
