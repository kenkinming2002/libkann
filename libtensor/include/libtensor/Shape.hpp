#pragma once

#include <array>
#include <tuple>
#include <vector>
#include <optional>

#include <stddef.h>
#include <assert.h>

namespace tensor
{
  struct Shape
  {
  public:
    static Shape from_vector(std::vector<size_t> dimensions)
    {
      Shape shape;
      shape.m_dimensions = std::move(dimensions);
      return shape;
    }

    static std::vector<size_t> to_vector(Shape shape)
    {
      return shape.m_dimensions;
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
    Shape subshape(size_t begin, size_t count) const
    {
      assert(begin+count<=m_dimensions.size());

      Shape result;
      for(size_t i=begin; i<begin+count; ++i)
        result.m_dimensions.push_back(m_dimensions[i]);
      return result;
    }

  public:
    static Shape make(const auto&... froms);

    Shape flatten(const auto&... hints)   const requires(sizeof...(hints)>0);
    Shape unflatten(const auto&... hints) const requires(sizeof...(hints)>0);

  public:
    Shape front(size_t count) const      { assert(rank()>=count); return this->subshape(0,     count); }
    Shape drop_front(size_t count) const { assert(rank()>=count); return this->subshape(count, rank() - count); }

    Shape back(size_t count) const      { assert(rank()>=count); return this->subshape(rank() - count, count); }
    Shape drop_back(size_t count) const { assert(rank()>=count); return this->subshape(0,              rank() - count);     }

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
    std::vector<size_t> m_dimensions;
  };

  Shape Shape::make(const auto&... froms)
  {
    Shape result;
    auto f = [&result](const auto& from) {
      if constexpr(std::is_convertible_v<std::remove_cvref_t<decltype(from)>, size_t>)
        result.m_dimensions.push_back(from);
      else if constexpr(std::is_same_v<std::remove_cvref_t<decltype(from)>, Shape>)
        for(size_t dimension : from.m_dimensions)
          result.m_dimensions.push_back(dimension);
      else
        []<bool flag=false>() { static_assert(flag, "Shape make only takes size_t or Shape"); }();
    };
    (f(froms), ...);
    return result;
  }

  struct Hint
  {
    enum class Type { SINGLE, SHAPE } type;
    std::optional<Shape> shape;

    static Hint single()                 { return Hint{.type = Type::SINGLE, .shape = std::nullopt}; }
    static Hint from_shape(Shape shape)  { return Hint{.type = Type::SHAPE,  .shape = std::move(shape)}; }
  };

  Shape Shape::flatten(const auto&... _hints)   const requires(sizeof...(_hints)>0)
  {
    const size_t N = sizeof...(_hints);

    std::array<Hint, N>   hints{_hints...};
    std::array<size_t, N> dimensions;

    size_t begin = 0;
    for(size_t i=0; i<N; ++i)
      switch(hints[i].type)
      {
      case Hint::Type::SINGLE:
        {
          dimensions[i] = this->dimension(begin);
          ++begin;
        }
        break;
      case Hint::Type::SHAPE:
        {
          Shape shape = *hints[i].shape;
          dimensions[i] = shape.size();
          begin += shape.rank();
        }
        break;
      }

    return std::apply([](auto... dimensions) { return Shape::make(dimensions...); }, dimensions);
  }

  Shape Shape::unflatten(const auto&... _hints) const requires(sizeof...(_hints)>0)
  {
    const size_t N = sizeof...(_hints);

    std::array<Hint, N>  hints{_hints...};
    std::array<Shape, N> shapes;

    size_t begin = 0;
    for(size_t i=0; i<N; ++i)
      switch(hints[i].type)
      {
      case Hint::Type::SINGLE:
        {
          shapes[i] = Shape::make(this->dimension(begin));
          ++begin;
        }
        break;
      case Hint::Type::SHAPE:
        {
          Shape shape = *hints[i].shape;
          shapes[i] = shape;
          begin += shape.rank();
        }
        break;
      }

    return std::apply([](auto... shapes) { return Shape::make(shapes...); }, shapes);
  }
}
