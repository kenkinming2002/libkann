#include <Eigen/Eigen>

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
}
