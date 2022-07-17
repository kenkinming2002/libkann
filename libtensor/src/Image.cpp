#include <libtensor/Image.hpp>

#include <variant>
#include <unsupported/Eigen/CXX11/Tensor>

namespace tensor
{
  // Tenary operator, but a and b need not have the same type
  inline static auto ternary(bool cond, auto a, auto b) noexcept -> std::variant<decltype(a), decltype(b)>
  {
    if(cond)
      return std::move(a);
    else
      return std::move(b);
  }

  template<typename T>
  static inline void eigen_gecorr2d(
      size_t M, size_t N, size_t K,
      size_t input_height, size_t input_width,
      size_t kernel_height, size_t kernel_width,
      size_t output_height, size_t output_width,
      const T* inputs, bool trans_inputs,
      const T* kernels, bool trans_kernels,
      T* outputs) noexcept
  {
    using TensorType = Eigen::Tensor<T, 4, Eigen::RowMajor>;
    using IndexType  = typename TensorType::Index;
    using Eigen::TensorMap;

    auto _inputs = ternary(trans_inputs,
      TensorMap<const TensorType>(inputs, (IndexType)K, (IndexType)M, (IndexType)input_height, (IndexType)input_width).shuffle(Eigen::array<int, 4>{1,0,2,3}),
      TensorMap<const TensorType>(inputs, (IndexType)M, (IndexType)K, (IndexType)input_height, (IndexType)input_width)
    );
    auto _kernels = ternary(trans_kernels,
      TensorMap<const TensorType>(kernels, (IndexType)N, (IndexType)K, (IndexType)kernel_height, (IndexType)kernel_width).shuffle(Eigen::array<int, 4>{1,0,2,3}),
      TensorMap<const TensorType>(kernels, (IndexType)K, (IndexType)N, (IndexType)kernel_height, (IndexType)kernel_width)
    );
    auto _outputs = TensorMap<TensorType>(outputs, (IndexType)M, (IndexType)N, (IndexType)output_height, (IndexType)output_width);

    auto padding_height = (kernel_height + output_height - input_height - 1) / 2;
    auto padding_width  = (kernel_width + output_width   - input_width  - 1) / 2;
    std::visit(
      [M, N, K, padding_height, padding_width, &outputs=_outputs](const auto& inputs, const auto& kernels)
      {
        outputs.setZero();
        for(size_t m=0; m<M; ++m)
          for(size_t n=0; n<N; ++n)
            for(size_t k=0; k<K; ++k)
            {
              auto input  = inputs .chip(m, 0).chip(k, 0);
              auto kernel = kernels.chip(k, 0).chip(n, 0);
              auto output = outputs.chip(m, 0).chip(n, 0);

              Eigen::array<std::pair<int, int>, 2> paddings;
              paddings[0] = std::make_pair(padding_height, padding_height);
              paddings[1] = std::make_pair(padding_width,  padding_width);
              auto padded_input = input.pad(paddings);

              const Eigen::array<ptrdiff_t, 2> dims({0, 1});
              output += padded_input.convolve(kernel, dims);
            }
      },
      _inputs, _kernels
    );
  }

  template<typename T>
  void eigen_geconv2d(
      size_t M, size_t N, size_t K,
      size_t input_height, size_t input_width,
      size_t kernel_height, size_t kernel_width,
      size_t output_height, size_t output_width,
      const T* inputs, bool trans_inputs,
      const T* kernels, bool trans_kernels,
      T* outputs) noexcept
  {
    using TensorType = Eigen::Tensor<T, 4, Eigen::RowMajor>;
    using IndexType  = typename TensorType::Index;
    using Eigen::TensorMap;

    auto _inputs = ternary(trans_inputs,
      TensorMap<const TensorType>(inputs, (IndexType)K, (IndexType)M, (IndexType)input_height, (IndexType)input_width).shuffle(Eigen::array<int, 4>{1,0,2,3}),
      TensorMap<const TensorType>(inputs, (IndexType)M, (IndexType)K, (IndexType)input_height, (IndexType)input_width)
    );
    auto _kernels = ternary(trans_kernels,
      TensorMap<const TensorType>(kernels, (IndexType)N, (IndexType)K, (IndexType)kernel_height, (IndexType)kernel_width).shuffle(Eigen::array<int, 4>{1,0,2,3}),
      TensorMap<const TensorType>(kernels, (IndexType)K, (IndexType)N, (IndexType)kernel_height, (IndexType)kernel_width)
    );
    auto _outputs = TensorMap<TensorType>(outputs, (IndexType)M, (IndexType)N, (IndexType)output_height, (IndexType)output_width);

    auto padding_height = (kernel_height + output_height - input_height - 1) / 2;
    auto padding_width  = (kernel_width + output_width   - input_width  - 1) / 2;
    std::visit(
      [M, N, K, padding_height, padding_width, &outputs=_outputs](const auto& inputs, const auto& kernels)
      {
        outputs.setZero();
        for(size_t m=0; m<M; ++m)
          for(size_t n=0; n<N; ++n)
            for(size_t k=0; k<K; ++k)
            {
              auto input  = inputs .chip(m, 0).chip(k, 0);
              auto kernel = kernels.chip(k, 0).chip(n, 0);
              auto output = outputs.chip(m, 0).chip(n, 0);

              Eigen::array<std::pair<int, int>, 2> paddings;
              paddings[0] = std::make_pair(padding_height, padding_height);
              paddings[1] = std::make_pair(padding_width,  padding_width);
              auto padded_input = input.pad(paddings);

              const Eigen::array<ptrdiff_t, 2> dims({0, 1});
              const Eigen::array<bool, 2> reverse({true, true});
              output += padded_input.convolve(kernel.reverse(reverse), dims);
            }
      },
      _inputs, _kernels
    );
  }

  template<typename T>
  void image2d_cross_correlate_raw(size_t M, size_t N, size_t K, Vec2 input_size, Vec2 kernel_size, Vec2 output_size, const T* inputs, bool trans_inputs, const T* kernels, bool trans_kernels, T* outputs) noexcept
  {
    eigen_gecorr2d(
        M, N, K,
        input_size.height(),   input_size.width(),
        kernel_size.height(),  kernel_size.width(),
        output_size.height(),  output_size.width(),
        inputs,  trans_inputs,
        kernels, trans_kernels,
        outputs
    );
  }

  template<typename T>
  void image2d_convolve_raw(size_t M, size_t N, size_t K, Vec2 input_size, Vec2 kernel_size, Vec2 output_size, const T* inputs, bool trans_inputs, const T* kernels, bool trans_kernels, T* outputs) noexcept
  {
    eigen_geconv2d(
        M, N, K,
        input_size.height(),  input_size.width(),
        kernel_size.height(), kernel_size.width(),
        output_size.height(), output_size.width(),
        inputs,  trans_inputs,
        kernels, trans_kernels,
        outputs
    );
  }

  template<typename T>
  Tensor<T> image2d_cross_correlate(Tensor<T> inputs, bool trans_inputs, Tensor<T> kernels, bool trans_kernels, Vec2 output_size)
  {
    auto M  = inputs .shape.dimension(0), K1 = inputs .shape.dimension(1);
    auto K2 = kernels.shape.dimension(0), N  = kernels.shape.dimension(1);

    if(trans_inputs)  std::swap(M, K1);
    if(trans_kernels) std::swap(K2, N);

    assert(K1 == K2);
    auto K = K1;

    auto input_size  = Vec2(inputs .shape.dimension(2), inputs .shape.dimension(3));
    auto kernel_size = Vec2(kernels.shape.dimension(2), kernels.shape.dimension(3));

    auto buffer_inputs  = inputs.buffer;
    auto buffer_kernels = kernels.buffer;
    auto buffer_outputs = std::make_shared<Buffer<T>>(M * N * output_size.height() * output_size.width());
    image2d_cross_correlate_raw(
        M, N, K,
        input_size, kernel_size, output_size,
        buffer_inputs ->data().data(), trans_inputs,
        buffer_kernels->data().data(), trans_kernels,
        buffer_outputs->data().data());

    return Tensor<T>(Shape::make(M, N, output_size.height(), output_size.width()), std::move(buffer_outputs));
  }

  template<typename T>
  Tensor<T> image2d_convolve(Tensor<T> inputs, bool trans_inputs, Tensor<T> kernels, bool trans_kernels, Vec2 output_size)
  {
    auto M  = inputs .shape.dimension(0), K1 = inputs .shape.dimension(1);
    auto K2 = kernels.shape.dimension(0), N  = kernels.shape.dimension(1);

    if(trans_inputs)  std::swap(M, K1);
    if(trans_kernels) std::swap(K2, N);

    assert(K1 == K2);
    auto K = K1;

    auto input_size  = Vec2(inputs .shape.dimension(2), inputs .shape.dimension(3));
    auto kernel_size = Vec2(kernels.shape.dimension(2), kernels.shape.dimension(3));

    auto buffer_inputs  = inputs.buffer;
    auto buffer_kernels = kernels.buffer;
    auto buffer_outputs = std::make_shared<Buffer<T>>(M * N * output_size.height() * output_size.width());
    image2d_convolve_raw(
        M, N, K,
        input_size, kernel_size, output_size,
        buffer_inputs ->data().data(), trans_inputs,
        buffer_kernels->data().data(), trans_kernels,
        buffer_outputs->data().data());

    return Tensor<T>(Shape::make(M, N, output_size.height(), output_size.width()), std::move(buffer_outputs));
  }

  template Tensor<float>       image2d_cross_correlate(Tensor<float>       inputs, bool trans_inputs, Tensor<float>       kernels, bool trans_kernels, Vec2 output_size);
  template Tensor<double>      image2d_cross_correlate(Tensor<double>      inputs, bool trans_inputs, Tensor<double>      kernels, bool trans_kernels, Vec2 output_size);
  template Tensor<long double> image2d_cross_correlate(Tensor<long double> inputs, bool trans_inputs, Tensor<long double> kernels, bool trans_kernels, Vec2 output_size);

  template Tensor<float>       image2d_convolve(Tensor<float>       inputs, bool trans_inputs, Tensor<float>       kernels, bool trans_kernels, Vec2 output_size);
  template Tensor<double>      image2d_convolve(Tensor<double>      inputs, bool trans_inputs, Tensor<double>      kernels, bool trans_kernels, Vec2 output_size);
  template Tensor<long double> image2d_convolve(Tensor<long double> inputs, bool trans_inputs, Tensor<long double> kernels, bool trans_kernels, Vec2 output_size);
}
