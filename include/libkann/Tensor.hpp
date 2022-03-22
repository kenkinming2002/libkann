#pragma once

#include <Eigen/Eigen>

#include <cereal/types/vector.hpp>

#include <memory>

namespace kann
{
  struct Tensor
  {
  public:
    static Tensor constant(size_t size, double value);

  public:
    Tensor() = default;
    Tensor(size_t size) : m_size(size)
    {
      m_values = std::make_unique_for_overwrite<double[]>(m_size);
    }

    Tensor(Eigen::VectorXd data);

  public:
    size_t size() const { return m_size; }

  public:
    double& asScalar()
    {
      assert(m_size == 1);
      return m_values[0];
    }

    const double& asScalar() const
    {
      assert(m_size == 1);
      return m_values[0];
    }

  public:
    auto asArray() &
    {
      return Eigen::ArrayXd::Map(m_values.get(), m_size);
    }

    auto asArray() const &
    {
      return Eigen::ArrayXd::Map(m_values.get(), m_size);
    }

  public:
    auto asVector() &
    {
      return Eigen::VectorXd::Map(m_values.get(), m_size);
    }

    auto asVector() const &
    {
      return Eigen::VectorXd::Map(m_values.get(), m_size);
    }

  public:
    auto asRowVector() &
    {
      return Eigen::RowVectorXd::Map(m_values.get(), m_size);
    }

    auto asRowVector() const &
    {
      return Eigen::RowVectorXd::Map(m_values.get(), m_size);
    }

  public:
    auto asMatrix(size_t rows, size_t cols) &
    {
      assert(m_size == rows * cols);
      return Eigen::MatrixXd::Map(m_values.get(), rows, cols);
    }

    auto asMatrix(size_t rows, size_t cols) const &
    {
      assert(m_size == rows * cols);
      return Eigen::MatrixXd::Map(m_values.get(), rows, cols);
    }

  public:
    template<typename BinaryOp>
    static Tensor binaryExpr(const Tensor& lhs, const Tensor& rhs, const BinaryOp& op)
    {
      assert(lhs.size() == rhs.size());

      Tensor result(lhs.size());
      result.asArray() = lhs.asArray().binaryExpr(rhs.asArray(), op);
      return result;
    }


  public:
    template<typename Archive>
    void save(Archive& archive) const
    {
      archive(cereal::make_size_tag(m_size));
      for(size_t i=0; i<m_size; ++i)
        archive(m_values[i]);
    }

    template<typename Archive>
    void load(Archive& archive)
    {
      archive(cereal::make_size_tag(m_size));
      m_values = std::make_unique_for_overwrite<double[]>(m_size);
      for(size_t i=0; i<m_size; ++i)
        archive(m_values[i]);
    }

  public:
    size_t m_size;
    std::unique_ptr<double[]> m_values;

  };

  inline Tensor Tensor::constant(size_t size, double value)
  {
    Tensor result(size);
    result.asArray().setConstant(value);
    return result;
  }

  inline Tensor::Tensor(Eigen::VectorXd v) : Tensor(v.size())
  {
    asVector() = v;
  }
}
