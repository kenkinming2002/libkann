#include <libkann/Tensor.hpp>

#include <Eigen/Eigen>

#include <range/v3/all.hpp>

namespace kann
{
  using EigenArray  = Eigen::ArrayXd;
  using EigenMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  template<typename Storage>
  static inline auto to_eigen_array(const TensorBase<Storage>& tensor)
  {
    return EigenArray::Map(tensor.data(), tensor.size());
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

  namespace utils
  {
    size_t max_coeff(TensorRef value)
    {
      size_t coeff;
      to_eigen_array(value).maxCoeff(&coeff);
      return coeff;
    }
  }

  namespace math
  {
    Tensor product(Tensor a, Tensor b, size_t M, size_t N, size_t K, bool transpose_a, bool transpose_b)
    {
      const auto& [shape_a1, shape_a2] = transpose_a ? a.shape().split(K, M) : a.shape().split(M, K);
      const auto& [shape_b1, shape_b2] = transpose_b ? b.shape().split(N, K) : b.shape().split(K, N);

      const auto& [shape_m, shape_k1] = transpose_a ? std::make_pair(shape_a2, shape_a1) :std::make_pair(shape_a1, shape_a2);
      const auto& [shape_k2, shape_n] = transpose_b ? std::make_pair(shape_b2, shape_b1) :std::make_pair(shape_b1, shape_b2);
      assert(shape_k1 == shape_k2);

      MutableTensor result = MutableTensor::create(Shape::concat(shape_m, shape_n));
      {
        auto _result = to_eigen_matrix(result.as_ref().reshape(Shape{shape_m.size(), shape_n.size()}));
        auto _a = to_eigen_matrix(a.as_ref().reshape(Shape{shape_a1.size(), shape_a2.size()}));
        auto _b = to_eigen_matrix(b.as_ref().reshape(Shape{shape_b1.size(), shape_b2.size()}));
        if(transpose_a)
        {
          if(transpose_b)
            _result.noalias() = _a.transpose() * _b.transpose();
          else
            _result.noalias() = _a.transpose() * _b;
        }
        else
        {
          if(transpose_b)
            _result.noalias() = _a * _b.transpose();
          else
            _result.noalias() = _a * _b;
        }
      }
      return result.as_const();
    }
  }
}
