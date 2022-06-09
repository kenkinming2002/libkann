#pragma once

#include <Eigen/Eigen>

#include <random>
#include <memory>

namespace kann
{
  struct Tensor
  {
  public:
    Tensor() = default;
    Tensor(size_t size) : m_size(size)
    {
      m_values = std::make_unique_for_overwrite<double[]>(m_size);
    }

  // Properties
  public:
    size_t size() const { return m_size; }

    double* data() { return m_values.get(); }
    const double* data() const { return m_values.get(); }

  public:
    double& operator[](size_t i)             { assert(i<size()); return m_values[i]; }
    const double& operator[](size_t i) const { assert(i<size()); return m_values[i]; }

  public:
    template<typename NullaryOp>
    static Tensor nullaryExpr(size_t size, const NullaryOp& op)
    {
      Tensor result(size);
      for(size_t i=0; i<size; ++i)
        result[i] = op();

      return result;
    }

    template<typename UnaryOp>
    static Tensor unaryExpr(const Tensor& value, const UnaryOp& op)
    {
      Tensor result(value.size());
      for(size_t i=0; i<value.size(); ++i)
        result[i] = op(value[i]);

      return result;
    }

    template<typename BinaryOp>
    static Tensor binaryExpr(const Tensor& lhs, const Tensor& rhs, const BinaryOp& op)
    {
      assert(lhs.size() == rhs.size());
      const size_t size = lhs.size();

      Tensor result(lhs.size());
      for(size_t i=0; i<size; ++i)
        result[i] = op(lhs[i], rhs[i]);

      return result;
    }

    template<typename TernaryOp>
    static Tensor ternaryExpr(const Tensor& arg1, const Tensor& arg2, const Tensor& arg3, const TernaryOp& op)
    {
      assert(arg1.size() == arg2.size() && arg2.size() == arg3.size());
      const size_t size = arg1.size();

      Tensor result(size);
      for(size_t i=0; i<size; ++i)
        result[i] = op(arg1[i], arg2[i], arg3[i]);

      return result;
    }

  public:
    static Tensor constant(size_t size, double value)
    {
      return nullaryExpr(size, [&](){
        return value;
      });
    }

    template<typename PRNG>
    static Tensor gaussian(size_t size, PRNG& prng, double mean, double variance)
    {
      std::normal_distribution<double> dist(mean, variance);
      return nullaryExpr(size, [&]() {
        return dist(prng);
      });
    }


  // Helpers
  public:
    double& asScalar()             { assert(m_size == 1); return m_values[0]; }
    const double& asScalar() const { assert(m_size == 1); return m_values[0]; }

  public:
    auto asArray()       & { return Eigen::ArrayXd::Map(data(), size()); }
    auto asArray() const & { return Eigen::ArrayXd::Map(data(), size()); }

  public:
    auto asVector()       & { return Eigen::VectorXd::Map(data(), size()); }
    auto asVector() const & { return Eigen::VectorXd::Map(data(), size()); }

  public:
    auto asRowVector()       & { return Eigen::RowVectorXd::Map(data(), size()); }
    auto asRowVector() const & { return Eigen::RowVectorXd::Map(data(), size()); }

  public:
    auto asMatrix(size_t rows, size_t cols)       & { assert(size() == rows * cols); return Eigen::MatrixXd::Map(data(), rows, cols); }
    auto asMatrix(size_t rows, size_t cols) const & { assert(size() == rows * cols); return Eigen::MatrixXd::Map(data(), rows, cols); }

  public:
    size_t m_size;
    std::unique_ptr<double[]> m_values;
  };
}
