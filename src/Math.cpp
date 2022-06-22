#include <libkann/Math.hpp>

#include <Eigen/Eigen>

#include <range/v3/all.hpp>

namespace kann::math
{
  using EigenArray  = Eigen::ArrayXf;
  using EigenVector = Eigen::RowVectorXf;
  using EigenMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  template<typename Storage>
  static inline auto to_eigen_array(const TensorBase<Storage>& tensor)
  {
    return EigenArray::Map(tensor.data(), tensor.size());
  }

  template<typename Storage>
  static inline auto to_eigen_vector(const TensorBase<Storage>& tensor)
  {
    assert(tensor.is_vector());
    return EigenVector::Map(tensor.data(), tensor.shape().size());
  }

  template<typename Storage>
  static inline auto to_eigen_matrix(const TensorBase<Storage>& tensor)
  {
    assert(tensor.is_matrix());
    return EigenMatrix::Map(tensor.data(),
      tensor.shape().dimension(0),
      tensor.shape().dimension(1)
    );
  }

  float norm(TensorRef value)
  {
    float sum = 0.0;
    for(size_t i=0; i<value.size(); ++i)
      sum += value.get(i) * value.get(i);
    return std::sqrt(sum);
  }

  void product(MutableTensorRef dst, TensorRef a, bool transpose_a, TensorRef b, bool transpose_b)
  {
    // Step 1: Compute all the shapes
    Shape M, N, K;
    {
      /* a.rank() = M.rank() + K.rank()
       * b.rank() = K.rank() + N.rank()
       * c.rank() = M.rank() + N.rank() */
      size_t rank_M = (a.rank() + dst.rank() - b.rank()) / 2;
      size_t rank_N = (b.rank() + dst.rank() - a.rank()) / 2;
      size_t rank_K = (a.rank() + b.rank() - dst.rank()) / 2;

      auto decompose = [](Shape shape, size_t rank1, size_t rank2, bool transpose)
      {
        Shape shape1, shape2;
        if(transpose)
          std::tie(shape2, shape1) = shape.split(rank2, rank1);
        else
          std::tie(shape1, shape2) = shape.split(rank1, rank2);
        return std::make_pair(shape1, shape2);
      };

      const auto& [_M1, _K1] = decompose(a.shape(), rank_M, rank_K, transpose_a);
      const auto& [_K2, _N1] = decompose(b.shape(), rank_K, rank_N, transpose_b);
      const auto& [_M2, _N2] = decompose(dst.shape(), rank_M, rank_N, false);

      assert(_M1 == _M2 && _N1 == _N2 && _K1 == _K2);
      std::tie(M, N, K) = std::make_tuple(_M1, _N1, _K1);
    }

    // Step 2: Reshape
    {
      auto reshape = [](auto&& value, Shape shape1, Shape shape2, bool transpose)
      {
        if(transpose)
          return std::forward<decltype(value)>(value).reshape(Shape(shape2.size(), shape1.size()));
        else
          return std::forward<decltype(value)>(value).reshape(Shape(shape1.size(), shape2.size()));
      };
      a   = reshape(a, M, K, transpose_a);
      b   = reshape(b, K, N, transpose_b);
      dst = reshape(dst, M, N, false);
    }

    // Step 3: Compute
    {
      auto _a   = to_eigen_matrix(a);
      auto _b   = to_eigen_matrix(b);
      auto _dst = to_eigen_matrix(dst);
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
  }

  static inline auto pad(Eigen::Ref<const EigenMatrix> matrix, Vec2 padding_size)
  {
    return EigenMatrix::NullaryExpr(matrix.rows() + 2 * padding_size.height(), matrix.cols() + 2 * padding_size.width(), [=](Eigen::Index row, Eigen::Index col)
    {
      if(static_cast<Eigen::Index>(padding_size.height()) <= row && row < matrix.rows() + static_cast<Eigen::Index>(padding_size.height())  &&
         static_cast<Eigen::Index>(padding_size.width())  <= col && col < matrix.cols() + static_cast<Eigen::Index>(padding_size.width()))
        return matrix(row - padding_size.height(), col - padding_size.width());

      return 0.0f;
    });
  }

  void image2d_operation(MutableTensorRef outputs, TensorRef inputs, bool transpose_inputs, TensorRef kernels, bool transpose_kernels, Image2DOperation operation)
  {
    // Step 1: Compute all the shapes, this is the same as in product() except we have to subtract the last two dimension
    Shape M, N, K, P, Q, R;
    {
      /* a.rank() = M.rank() + K.rank()
       * b.rank() = K.rank() + N.rank()
       * c.rank() = M.rank() + N.rank() */
      size_t rank_M = (inputs.rank()  + outputs.rank() - kernels.rank()) / 2 - 1;
      size_t rank_N = (kernels.rank() + outputs.rank() - inputs.rank() ) / 2 - 1;
      size_t rank_K = (inputs.rank()  + kernels.rank() - outputs.rank()) / 2 - 1;

      auto decompose = [](Shape shape, size_t rank1, size_t rank2, bool transpose)
      {
        Shape shape1, shape2;
        if(transpose)
          std::tie(shape2, shape1) = shape.drop_back(2).split(rank2, rank1);
        else
          std::tie(shape1, shape2) = shape.drop_back(2).split(rank1, rank2);
        return std::make_tuple(shape1, shape2, shape.back(2));
      };

      Shape _M1, _M2;
      Shape _N1, _N2;
      Shape _K1, _K2;

      std::tie(_M1, _K1, P) = decompose(inputs.shape(),  rank_M, rank_K, transpose_inputs);
      std::tie(_K2, _N1, Q) = decompose(kernels.shape(), rank_K, rank_N, transpose_kernels);
      std::tie(_M2, _N2, R) = decompose(outputs.shape(), rank_M, rank_N, false);
      assert(_M1 == _M2 && _N1 == _N2 && _K1 == _K2);
      std::tie(M, N, K) = std::make_tuple(_M1, _N1, _K1);
    }

    // Step 2: Reshape
    {
      auto reshape = [](auto&& value, Shape shape1, Shape shape2, Shape shape3, bool transpose)
      {
        if(transpose)
          return std::forward<decltype(value)>(value).reshape(Shape::concat(Shape(shape2.size(), shape1.size()), shape3));
        else
          return std::forward<decltype(value)>(value).reshape(Shape::concat(Shape(shape1.size(), shape2.size()), shape3));
      };
      inputs  = reshape(inputs,  M, K, P, transpose_inputs);
      kernels = reshape(kernels, K, N, Q, transpose_kernels);
      outputs = reshape(outputs, M, N, R, false);

    }

    // Step 3: Compute
    {
      outputs.fill(0.0);
      for(size_t i = 0; i<M.size(); ++i)
        for(size_t j = 0; j<N.size(); ++j)
          for(size_t k = 0; k<K.size(); ++k)
          {
            TensorRef input  = transpose_inputs  ? inputs[k][i]  : inputs[i][k];
            TensorRef kernel = transpose_kernels ? kernels[j][k] : kernels[k][j];
            MutableTensorRef output = outputs[i][j];

            const Vec2 input_size  = Vec2(input.dimension(0), input.dimension(1));
            const Vec2 kernel_size = Vec2(kernel.dimension(0), kernel.dimension(1));
            const Vec2 output_size = Vec2(output.dimension(0), output.dimension(1));
            const Vec2 padding_size = ((output_size - input_size) + (kernel_size - Vec2(1,1))) / 2;

            auto _input = to_eigen_matrix(input);
            auto _kernel = to_eigen_matrix(kernel);
            auto _output = to_eigen_matrix(output);
            auto _padded_input = pad(_input, padding_size);

            switch(operation)
            {
            case Image2DOperation::CONVOLUTION:
              _output.noalias() += EigenMatrix::NullaryExpr(_output.rows(), _output.cols(), [&](Eigen::Index row, Eigen::Index col) {
                return pad(_input, padding_size).block(row, col, _kernel.rows(), _kernel.cols()).cwiseProduct(_kernel.reverse()).sum();
              });
              break;
            case Image2DOperation::CROSS_CORRELATION:
              _output.noalias() += EigenMatrix::NullaryExpr(_output.rows(), _output.cols(), [&](Eigen::Index row, Eigen::Index col) {
                return pad(_input, padding_size).block(row, col, _kernel.rows(), _kernel.cols()).cwiseProduct(_kernel).sum();
              });
              break;
            }
          }
    }
  }
}
