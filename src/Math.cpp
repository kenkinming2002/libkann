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

  static inline void operation_impl(Operation operation, const float& from, float& to, float value)
  {
    switch(operation)
    {
    case Operation::STORE:
      to = from;
      break;
    case Operation::FMA:
      to = to + value * from;
      break;
    default:
      assert(false && "Unreachable");
    }
  }

  void broadcast(TensorRef from, MutableTensorRef to, Operation operation, Direction direction, double value)
  {
    // Step 1: Compute shape
    Shape left, right;
    switch(direction)
    {
    case Direction::LEFT:  std::tie(left, right) = to.shape().split(to.rank() - from.rank(), from.rank()); assert(from.shape() == right); break;
    case Direction::RIGHT: std::tie(left, right) = to.shape().split(from.rank(), to.rank() - from.rank()); assert(from.shape() == left); break;
    }

    // Step 2: Reshape
    from = from.reshape(Shape(from.size()));
    to = to.reshape(Shape(left.size(), right.size()));

    // Step 3: Compute, hopefully the compiler is able to vectorize through this
    for(size_t i=0; i<to.dimension(0); ++i)
      for(size_t j=0; j<to.dimension(1); ++j)
        switch(direction)
        {
        case Direction::LEFT:  operation_impl(operation, from[j].get(0), to[i][j].get(0), value); break;
        case Direction::RIGHT: operation_impl(operation, from[i].get(0), to[i][j].get(0), value); break;
        }
  }

  void reduce(TensorRef from, MutableTensorRef to, Operation operation, Direction direction, double value)
  {
    // Step 1: Compute shape
    Shape left, right;
    switch(direction)
    {
    case Direction::LEFT:  std::tie(left, right) = from.shape().split(from.rank() - to.rank(), to.rank()); assert(to.shape() == right); break;
    case Direction::RIGHT: std::tie(left, right) = from.shape().split(to.rank(), from.rank() - to.rank()); assert(to.shape() == left); break;
    }

    // Step 2: Reshape
    from = from.reshape(Shape(left.size(), right.size()));
    to = to.reshape(Shape(to.size()));

    // Step 3: Compute, hopefully the compiler is able to vectorize through this
    for(size_t i=0; i<from.dimension(0); ++i)
      for(size_t j=0; j<from.dimension(1); ++j)
        switch(direction)
        {
        case Direction::LEFT:  operation_impl(operation, from[i][j].get(0), to[j].get(0), value); break;
        case Direction::RIGHT: operation_impl(operation, from[i][j].get(0), to[i].get(0), value); break;
        }
  }

  void transform(TensorRef from, MutableTensorRef to, Operation operation, double value)
  {
    assert(from.shape() == to.shape());
    from = from.reshape(Shape(from.size()));
    to   = to.reshape(Shape(to.size()));
    for(size_t i=0; i<from.size(); ++i)
      operation_impl(operation, from[i].get(0), to[i].get(0), value);
  }


  void product(TensorRef a, bool transpose_a, TensorRef b, bool transpose_b, MutableTensorRef c)
  {
    // Step 1: Compute all the shapes
    Shape M, N, K;
    {
      /* a.rank() = M.rank() + K.rank()
       * b.rank() = K.rank() + N.rank()
       * c.rank() = M.rank() + N.rank() */
      size_t rank_M = (a.rank() + c.rank() - b.rank()) / 2;
      size_t rank_N = (b.rank() + c.rank() - a.rank()) / 2;
      size_t rank_K = (a.rank() + b.rank() - c.rank()) / 2;

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
      const auto& [_M2, _N2] = decompose(c.shape(), rank_M, rank_N, false);

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
      a = reshape(a, M, K, transpose_a);
      b = reshape(b, K, N, transpose_b);
      c = reshape(c, M, N, false);
    }

    // Step 3: Compute
    {
      auto _a = to_eigen_matrix(a);
      auto _b = to_eigen_matrix(b);
      auto _c = to_eigen_matrix(c);
      if(transpose_a)
      {
        if(transpose_b)
          _c.noalias() = _a.transpose() * _b.transpose();
        else
          _c.noalias() = _a.transpose() * _b;
      }
      else
      {
        if(transpose_b)
          _c.noalias() = _a * _b.transpose();
        else
          _c.noalias() = _a * _b;
      }
    }
  }

  // Get the shapes from ranks
  static inline Tensor generic_tensor_product(TensorRef a, TensorRef b, size_t rank_m, size_t rank_n, size_t rank_k, bool transpose_a, bool transpose_b, auto shape_impl, auto impl)
  {
    // Step 1: Compute all the shapes
    Shape M, N, K, P, Q, R;
    {
      auto decompose = [](Shape shape, size_t rank1, size_t rank2, bool transpose) {
        Shape shape1, shape2;
        if(transpose)
          std::tie(shape2, shape1) = shape.front(rank1+rank2).split(rank2, rank1);
        else
          std::tie(shape1, shape2) = shape.front(rank1+rank2).split(rank1, rank2);

        Shape shape3 = shape.drop_front(rank1+rank2);
        return std::make_tuple(shape1, shape2, shape3);
      };
      Shape _K1, _K2;
      std::tie(M, _K1, P) = decompose(a.shape(), rank_m, rank_k, transpose_a);
      std::tie(_K2, N, Q) = decompose(b.shape(), rank_k, rank_n, transpose_b);

      assert(_K1 == _K2);
      K = _K1;

      R = shape_impl(P, Q);
    }

    // Step 3: Create the result tensor and obtain a mutable reference to it
    MutableTensor result = MutableTensor::create(Shape::concat(M, N, R));
    result.fill(0.0);

    MutableTensorRef c = result.as_ref();

    // Step 3: Reshape to flatten the first two dimension
    {
      auto reshape = [](auto&& value, Shape shape1, Shape shape2, Shape shape3, bool transpose) {
        if(transpose)
          return std::forward<decltype(value)>(value).reshape(Shape::concat(Shape(shape2.size(), shape1.size()), shape3));
        else
          return std::forward<decltype(value)>(value).reshape(Shape::concat(Shape(shape1.size(), shape2.size()), shape3));
      };
      a = reshape(a, M, K, P, transpose_a);
      b = reshape(b, K, N, Q, transpose_b);
      c = reshape(c, M, N, R, false);
    }

    // Step 4: Call the supplied impl function
    {
      for(size_t i = 0; i<M.size(); ++i)
        for(size_t j = 0; j<N.size(); ++j)
          for(size_t k = 0; k<K.size(); ++k)
          {
            TensorRef a_elem = transpose_a ? a[k][i] : a[i][k];
            TensorRef b_elem = transpose_b ? b[j][k] : b[k][j];
            MutableTensorRef c_elem = c[i][j];
            impl(a_elem, b_elem, c_elem);
          }
    }
    return result.as_const();
  }

  static inline Shape get_output_shape(Shape input_shape, Shape kernel_shape, Vec2 padding_size)
  {
    assert(input_shape.rank() == 2);
    assert(kernel_shape.rank() == 2);
    Vec2 input_size  = Vec2(input_shape.dimension(0), input_shape.dimension(1));
    Vec2 kernel_size = Vec2(kernel_shape.dimension(0), kernel_shape.dimension(1));
    Vec2 output_size = (input_size + 2 * padding_size) - (kernel_size - Vec2(1,1));
    return Shape(output_size.height(), output_size.width());
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

  static inline void cross_correlate2d_impl(TensorRef input, TensorRef kernel, MutableTensorRef output, Vec2 padding_size)
  {
    auto _input = to_eigen_matrix(input);
    auto _kernel = to_eigen_matrix(kernel);
    auto _output = to_eigen_matrix(output);
    _output.noalias() += EigenMatrix::NullaryExpr(_output.rows(), _output.cols(), [&](Eigen::Index row, Eigen::Index col) {
      return pad(_input, padding_size).block(row, col, _kernel.rows(), _kernel.cols()).cwiseProduct(_kernel).sum();
    });
  }

  static inline void convolve2d_impl(TensorRef input, TensorRef kernel, MutableTensorRef output, Vec2 padding_size)
  {
    auto _input = to_eigen_matrix(input);
    auto _kernel = to_eigen_matrix(kernel);
    auto _output = to_eigen_matrix(output);
    _output.noalias() += EigenMatrix::NullaryExpr(_output.rows(), _output.cols(), [&](Eigen::Index row, Eigen::Index col) {
      return pad(_input, padding_size).block(row, col, _kernel.rows(), _kernel.cols()).cwiseProduct(_kernel.reverse()).sum();
    });
  }

  Tensor cross_correlate2d(Tensor inputs, Tensor kernels,
      size_t rank_m, size_t rank_n, size_t rank_k,
      bool transpose_input, bool transpose_kernel,
      Vec2 padding_size)
  {
    using namespace std::placeholders;
    return generic_tensor_product(inputs.as_ref(), kernels.as_ref(),
        rank_m, rank_n, rank_k,
        transpose_input, transpose_kernel,
        std::bind(&get_output_shape, _1, _2, padding_size),
        std::bind(&cross_correlate2d_impl, _1, _2, _3, padding_size));
  }

  Tensor convolve2d(Tensor inputs, Tensor kernels, size_t rank_m, size_t rank_n, size_t rank_k, bool transpose_input, bool transpose_kernel, Vec2 padding_size)
  {
    using namespace std::placeholders;
    return generic_tensor_product(inputs.as_ref(), kernels.as_ref(),
        rank_m, rank_n, rank_k,
        transpose_input, transpose_kernel,
        std::bind(&get_output_shape, _1, _2, padding_size),
        std::bind(&convolve2d_impl, _1, _2, _3, padding_size));
  }

  Tensor add(Tensor a, Tensor b)
  {
    return cwise(std::move(a), std::move(b), [](float a, float b){
      return a+b;
    });
  }

  Tensor sub(Tensor a, Tensor b)
  {
    return cwise(std::move(a), std::move(b), [](float a, float b){
      return a-b;
    });
  }

  Tensor scale(Tensor a, float factor)
  {
    return cwise(std::move(a), [&](float a){
      return a * factor;
    });
  }
}
