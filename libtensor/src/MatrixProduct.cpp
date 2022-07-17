#include <libtensor/MatrixProduct.hpp>

#include <variant>
#include <Eigen/Eigen>

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
  static inline void eigen_gemm(size_t M, size_t N, size_t K, const T* A, bool trans_A, const T* B, bool trans_B, T* C) noexcept
  {
    using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    auto _A = ternary(trans_A,
      Eigen::Map<const MatrixType>(A, K, M).transpose(),
      Eigen::Map<const MatrixType>(A, M, K)
    );

    auto _B = ternary(trans_B,
      Eigen::Map<const MatrixType>(B, N, K).transpose(),
      Eigen::Map<const MatrixType>(B, K, N)
    );

    auto _C = Eigen::Map<MatrixType>(C, M, N);

    std::visit([&C=_C](const auto& A, const auto& B) { C = A * B; }, _A, _B);
  }

  template<typename T>
  void matrix_product_raw(size_t M, size_t N, size_t K, const T* A, bool trans_A, const T* B, bool trans_B, T* C) noexcept
  {
    eigen_gemm(M, N, K, A, trans_A, B, trans_B, C);
  }

  template<typename T>
  Tensor<T> matrix_product(Tensor<T> a, bool trans_a, Tensor<T> b, bool trans_b)
  {
    auto [M, K1] = std::make_pair(a.shape.dimension(0), a.shape.dimension(1));
    auto [K2, N] = std::make_pair(b.shape.dimension(0), b.shape.dimension(1));

    if(trans_a) std::swap(M, K1);
    if(trans_b) std::swap(K2, N);
    if(K1 != K2) throw std::runtime_error("K mismatch");
    size_t K = K1;

    auto buffer_a = a.buffer;
    auto buffer_b = b.buffer;
    auto buffer_c = std::make_shared<Buffer<T>>(M*N);
    matrix_product_raw(M, N, K, buffer_a->data().data(), trans_a, buffer_b->data().data(), trans_b, buffer_c->data().data());
    return Tensor<T>(Shape::make(M, N), std::move(buffer_c));
  }

  template Tensor<float>  matrix_product(Tensor<float>  a, bool trans_a, Tensor<float>  b, bool trans_b);
  template Tensor<double> matrix_product(Tensor<double> a, bool trans_a, Tensor<double> b, bool trans_b);
  template Tensor<long double> matrix_product(Tensor<long double> a, bool trans_a, Tensor<long double> b, bool trans_b);
}

