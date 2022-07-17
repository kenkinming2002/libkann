#include <libtensor/MatrixProduct.hpp>

#include <Eigen/Eigen>

namespace tensor
{
  // Eigen API wraped in BLAS like API
  template<typename T>
  using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  template<typename T>
  static inline void _eigen_gemm(const Eigen::Ref<const MatrixType<T>> A, const Eigen::Ref<const MatrixType<T>> B, Eigen::Ref<MatrixType<T>> C)
  {
    C = A * B;
  }

  template<typename T>
  static inline void eigen_gemm(size_t M, size_t N, size_t K, const T* A, bool trans_A, const T* B, bool trans_b, T* C)
  {
    // Blas API support transposed arguement directly so we do not need to do this weird dance
    if(trans_A)
    {
      if(trans_b)
        _eigen_gemm<T>(Eigen::Map<const MatrixType<T>>(A, K, M).transpose(), Eigen::Map<const MatrixType<T>>(B, N, K).transpose(), Eigen::Map<MatrixType<T>>(C, M, N));
      else
        _eigen_gemm<T>(Eigen::Map<const MatrixType<T>>(A, K, M).transpose(), Eigen::Map<const MatrixType<T>>(B, K, N),             Eigen::Map<MatrixType<T>>(C, M, N));
    }
    else
    {
      if(trans_b)
        _eigen_gemm<T>(Eigen::Map<const MatrixType<T>>(A, M, K), Eigen::Map<const MatrixType<T>>(B, N, K).transpose(), Eigen::Map<MatrixType<T>>(C, M, N));
      else
        _eigen_gemm<T>(Eigen::Map<const MatrixType<T>>(A, M, K), Eigen::Map<const MatrixType<T>>(B, K, N),             Eigen::Map<MatrixType<T>>(C, M, N));
    }
  }

  template<typename T>
  void matrix_product_raw(size_t M, size_t N, size_t K, const T* A, bool trans_A, const T* B, bool trans_B, T* C)
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

