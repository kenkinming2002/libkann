#pragma once

#include <libkann/Shape.hpp>

#include <range/v3/all.hpp>

#include <random>
#include <memory>
#include <tuple>

namespace kann
{
  // Reference to a tensor
  template<typename T>
  struct TensorRef
  {
  public:
    constexpr TensorRef(Shape shape, std::span<T> data)
      : m_shape(std::move(shape)), m_data(data)
    {
      assert(m_shape.size() == m_data.size());
    }

  public:
    constexpr size_t dimension(size_t i) const { return m_shape.dimension(i); }
    constexpr size_t rank() const { return m_shape.rank(); }
    constexpr const Shape& shape() const { return m_shape; }

  public:
    bool is_scalar() const { return m_shape.is_scalar(); }
    bool is_vector() const { return m_shape.is_vector(); }
    bool is_matrix() const { return m_shape.is_matrix(); }

  public:
    T& as_scalar() const { assert(is_scalar()); return *data(); }

  public:
    constexpr T* data() const { return m_data.data(); }
    constexpr size_t size()   const { return m_data.size(); }

  public:
    constexpr void fill(T value)
    {
      ranges::fill(m_data, value);
    }

    template<typename PRNG>
    constexpr void fill_uniform(PRNG& prng, T a, T b)
    {
      std::uniform_real_distribution<T> dist(a, b);
      ranges::generate(m_data, [&]() { return dist(prng); });
    }

    template<typename PRNG>
    constexpr void fill_normal(PRNG& prng, T mean, T stddev)
    {
      std::normal_distribution<T> dist(mean, stddev);
      ranges::generate(m_data, [&]() { return dist(prng); });
    }

  public:
    constexpr TensorRef<T> operator[](size_t i) const
    {
      const auto& [left, right] = m_shape.split(1, m_shape.rank() - 1);

      const size_t dimension = left.size();
      const size_t stride = right.size();

      assert(i < dimension);
      return TensorRef(right, m_data.subspan(i * stride, stride));
    }

    constexpr TensorRef<T> reshape(Shape new_shape) const
    {
      assert(m_shape.size() == new_shape.size());
      return TensorRef(std::move(new_shape), m_data);
    }

    constexpr TensorRef<T> flatten() const
    {
      return reshape(Shape(size()));
    }

  private:
    Shape m_shape;
    std::span<T> m_data;
  };

  // Real tensor
  template<typename T>
  struct Tensor
  {
  public:
    constexpr Tensor(Shape shape, std::unique_ptr<T[]> data)
      : m_shape(std::move(shape)), m_data(std::move(data)) {}

  public:
    static Tensor<T> create(Shape shape)
    {
      std::unique_ptr<T[]> data = std::make_unique_for_overwrite<T[]>(shape.size());
      return Tensor(std::move(shape), std::move(data));
    }

  public:
    Tensor reshape(Shape shape) &&
    {
      assert(shape.size() == m_shape.size());
      return Tensor(shape, std::move(m_data));
    }

  public:
    Tensor clone() const
    {
      Tensor result = create(m_shape);
      ranges::copy_n(m_data.get(), m_shape.size(), result.m_data.get());
      return result;
    }

  public:
    auto as_ref()             -> TensorRef<T>       { return TensorRef<T>(m_shape,       std::span(m_data.get(), m_shape.size())); }
    auto as_ref()       const -> TensorRef<const T> { return TensorRef<const T>(m_shape, std::span(m_data.get(), m_shape.size())); }
    auto as_const_ref() const -> TensorRef<const T> { return TensorRef<const T>(m_shape, std::span(m_data.get(), m_shape.size())); }

  public:
    double& operator()(auto... indices)             { return m_data[m_shape.get_index(std::array<size_t, sizeof...(indices)>{indices...})]; }
    const double& operator()(auto... indices) const { return m_data[m_shape.get_index(std::array<size_t, sizeof...(indices)>{indices...})]; }

  private:
    Shape m_shape;
    std::unique_ptr<T[]> m_data;
  };
}
