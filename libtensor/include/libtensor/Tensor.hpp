#pragma once

#include <libtensor/Shape.hpp>

#include <range/v3/all.hpp>

#include <random>
#include <memory>
#include <tuple>

namespace tensor
{
  template<typename T>
  struct Tensor
  {
  public:
    Tensor(Shape shape, std::shared_ptr<T[]> data)
      : m_shape(std::move(shape)), m_data(std::move(data)) {}

  public:
    operator Tensor<const T>() const requires(!std::is_const_v<T>)
    {
      return Tensor<const T>(m_shape, m_data);
    }

  public:
    static Tensor<T> create(Shape shape)
    {
      std::unique_ptr<T[]> data = std::make_unique_for_overwrite<T[]>(shape.size());
      return Tensor(std::move(shape), std::move(data));
    }

    Tensor<std::remove_const_t<T>> clone() const
    {
      Tensor<std::remove_const_t<T>> result = Tensor<std::remove_const_t<T>>::create(m_shape);
      std::copy_n(data(), size(), result.data());
      return result;
    }

  public:
    constexpr T* data()     const { return m_data.get(); }
    constexpr size_t size() const { return m_shape.size(); }

  public:
    constexpr const Shape& shape() const { return m_shape; }
    constexpr size_t dimension(size_t i) const { return m_shape.dimension(i); }
    constexpr size_t rank() const { return m_shape.rank(); }

  public:
    Tensor reshape(Shape shape) const
    {
      assert(shape.size() == m_shape.size());
      return Tensor(shape, m_data);
    }

    Tensor flatten() const
    {
      return reshape(Shape(size()));
    }

  public:
    void fill(T value)
    {
      ranges::fill_n(this->data(), this->size(), value);
    }

    template<typename PRNG>
    void fill_uniform(PRNG& prng, T a, T b)
    {
      std::uniform_real_distribution<T> dist(a, b);
      ranges::generate_n(this->data(), this->size(), [&]() { return dist(prng); });
    }

    template<typename PRNG>
    void fill_normal(PRNG& prng, T mean, T stddev)
    {
      std::normal_distribution<T> dist(mean, stddev);
      ranges::generate_n(this->data(), this->size(), [&]() { return dist(prng); });
    }

  public:
    T& operator()(auto... indices)             { return m_data[m_shape.get_index(std::array<size_t, sizeof...(indices)>{indices...})]; }
    const T& operator()(auto... indices) const { return m_data[m_shape.get_index(std::array<size_t, sizeof...(indices)>{indices...})]; }

  private:
    Shape m_shape;
    std::shared_ptr<T[]> m_data;
  };
}
