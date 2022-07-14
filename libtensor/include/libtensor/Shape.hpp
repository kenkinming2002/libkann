#pragma once

#include <libtensor/StaticVector.hpp>

#include <vector>

#include <stddef.h>
#include <assert.h>

namespace tensor
{
  struct FlattenSingle {};
  static constexpr FlattenSingle flatten_single = {};

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
    template<size_t N>
    static constexpr auto concat(std::array<Shape, N> shapes)
    {
      Shape result;
      for(const auto& shape : shapes)
        for(const auto& dimension : shape.m_dimensions)
          result.m_dimensions.push_back(dimension);

      return result;
    }

    template<size_t N>
    static constexpr std::array<Shape, N> split(Shape shape, std::array<size_t, N> ranks)
    {
      std::array<Shape, N> shapes;
      size_t begin = 0;
      for(size_t i=0; i<ranks.size(); ++i)
      {
        begin += ranks[i];
        shapes[i] = shape.subshape(begin, ranks[i]);
      }
      return shapes;
    }

  public:
    constexpr Shape flatten(const auto&... hints) const requires(sizeof...(hints)>0)
    {
      // Hints could a size_t or shape
      auto size_from_hint = [shape=*this](auto hint) mutable -> size_t {
        size_t rank;
        if constexpr(std::is_same_v<std::remove_cvref_t<decltype(hint)>, FlattenSingle>)
          rank = 1;
        else if constexpr(std::is_same_v<std::remove_cvref_t<decltype(hint)>, Shape>)
          rank = hint.rank();
        else
          []<bool flag=false>() { static_assert(flag, "Unsupported hint type"); }();

        size_t size = shape.front(rank).size();
        shape = shape.drop_front(rank);
        return size;
      };

      // Enforce order of evaluation by using brace-initializer
      std::array<size_t, sizeof...(hints)> sizes{size_from_hint(hints)...};
      return std::apply([](auto...sizes) { return Shape{sizes...}; }, sizes);
    }

    constexpr Shape unflatten(const auto&... hints) const requires(sizeof...(hints)>0)
    {
      // Hints could a size_t or shape
      auto shape_from_hint = [shape=*this](auto hint) mutable -> Shape {
        Shape result;
        if constexpr(std::is_same_v<std::remove_cvref_t<decltype(hint)>, FlattenSingle>)
          result = shape.front(1);
        else if constexpr(std::is_same_v<std::remove_cvref_t<decltype(hint)>, Shape>)
          result = hint;
        else
          []<bool flag=false>() { static_assert(flag, "Unsupported hint type"); }();

        shape = shape.drop_front(1);
        return result;
      };

      // Enforce order of evaluation by using brace-initializer
      std::array<Shape, sizeof...(hints)> shapes{shape_from_hint(hints)...};
      return std::apply([](auto...shapes) { return Shape::concat<sizeof...(shapes)>({shapes...}); }, shapes);
    }

  public:
    constexpr Shape subshape(size_t begin, size_t count) const
    {
      assert(begin+count<=m_dimensions.size());

      Shape result;
      for(size_t i=begin; i<begin+count; ++i)
        result.m_dimensions.push_back(m_dimensions[i]);
      return result;
    }

  public:
    constexpr Shape front(size_t count) const      { assert(rank()>=count); return this->subshape(0,     count); }
    constexpr Shape drop_front(size_t count) const { assert(rank()>=count); return this->subshape(count, rank() - count); }

    constexpr Shape back(size_t count) const      { assert(rank()>=count); return this->subshape(rank() - count, count); }
    constexpr Shape drop_back(size_t count) const { assert(rank()>=count); return this->subshape(0,              rank() - count);     }

  public:
    template<size_t N>
    size_t get_index(std::array<size_t, N> indices) const
    {
      assert(rank() == N);
      std::array<size_t, N> dimensions = [this]<std::size_t... Is>(std::index_sequence<Is...>) { return std::array{m_dimensions[Is]...}; }(std::make_index_sequence<N>());
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
