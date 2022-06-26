#pragma once

#include <libkann/StaticVector.hpp>

#include <vector>

#include <stddef.h>
#include <assert.h>

namespace kann
{
  struct Shape
  {
  public:
    static constexpr size_t MAX_DIMENSION = 8;

  public:
    static Shape from_vector(std::vector<size_t> dimensions)
    {
      Shape shape;
      for(size_t dimension : dimensions)
        shape.m_dimensions.push_back(dimension);
      return shape;
    }

    static std::vector<size_t> to_vector(Shape shape)
    {
      return shape.m_dimensions | ranges::to_vector;
    }

  public:
    constexpr Shape() = default;
    template<typename... Dimensions>
    constexpr explicit Shape(Dimensions... dimensions)
    {
      (m_dimensions.push_back(dimensions), ...);
    }

  public:
    constexpr size_t rank() const { return m_dimensions.size(); }
    constexpr size_t dimension(size_t i) const { return m_dimensions[i]; }

  public:
    constexpr size_t size() const
    {
      size_t size = 1;
      for(size_t i=0; i<rank(); ++i)
        size *= dimension(i);
      return size;
    }

  public:
    bool is_scalar() const { return rank() == 0; }
    bool is_vector() const { return rank() == 1; }
    bool is_matrix() const { return rank() == 2; }

  public:
    constexpr std::pair<Shape, Shape> split(size_t rank1, size_t rank2) const
    {
      assert(rank1+rank2==rank());
      Shape result1, result2;

      for(size_t i=0; i<rank1; ++i)
        result1.m_dimensions.push_back(dimension(i));

      for(size_t i=rank1; i<rank1+rank2; ++i)
        result2.m_dimensions.push_back(dimension(i));

      return std::make_pair(result1, result2);
    }

    constexpr std::tuple<Shape, Shape, Shape> split(size_t rank1, size_t rank2, size_t rank3) const
    {
      assert(rank1+rank2+rank3==rank());
      Shape result1, result2, result3;

      for(size_t i=0; i<rank1; ++i)
        result1.m_dimensions.push_back(dimension(i));

      for(size_t i=rank1; i<rank1+rank2; ++i)
        result2.m_dimensions.push_back(dimension(i));

      for(size_t i=rank1+rank2; i<rank1+rank2+rank3; ++i)
        result3.m_dimensions.push_back(dimension(i));

      return std::make_tuple(result1, result2, result3);
    }

    static constexpr Shape concat(Shape a, Shape b)
    {
      Shape shape;
      for(size_t dimension : a.m_dimensions)
        shape.m_dimensions.push_back(dimension);

      for(size_t dimension : b.m_dimensions)
        shape.m_dimensions.push_back(dimension);

      return shape;
    }

    static constexpr Shape concat(Shape a, Shape b, Shape c)
    {
      Shape shape;
      for(size_t dimension : a.m_dimensions)
        shape.m_dimensions.push_back(dimension);

      for(size_t dimension : b.m_dimensions)
        shape.m_dimensions.push_back(dimension);

      for(size_t dimension : c.m_dimensions)
        shape.m_dimensions.push_back(dimension);

      return shape;
    }

  public:
    constexpr Shape front(size_t count) const      { assert(rank()>=count); return this->split(count, rank() - count).first; }
    constexpr Shape drop_front(size_t count) const { assert(rank()>=count); return this->split(count, rank() - count).second; }

    constexpr Shape back(size_t count) const      { assert(rank()>=count); return this->split(rank() - count, count).second; }
    constexpr Shape drop_back(size_t count) const { assert(rank()>=count); return this->split(rank() - count, count).first; }

  public:
    template<size_t N>
    size_t get_index(std::array<size_t, N> indices) const
    {
      assert(rank() == N);
      std::array<size_t, N> dimensions = [this]<std::size_t... Is>(std::index_sequence<Is>...) { return std::array{m_dimensions[Is]...}; }(std::make_index_sequence<N>());
      std::array<size_t, N> strides;
      for(size_t i=0; i<N; ++i)
      {
        strides[i] = 1;
        for(size_t j=i+1; j<N; ++j)
          strides[i] *= dimensions[j];
      }

      size_t index = 0;
      for(size_t i=0; i<N; ++i)
      {
        assert(indices[i] < dimensions[i]);
        index += indices[i] * strides[i];
      }

      return index;
    }

  public:
    friend bool operator==(const Shape& lhs, const Shape& rhs) = default;
    friend bool operator!=(const Shape& lhs, const Shape& rhs) = default;

  private:
    // TODO: Use fixed size vector
    StaticVector<size_t, MAX_DIMENSION> m_dimensions;
  };
}
