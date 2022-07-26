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
    std::vector<size_t> dimensions;

  public:
    static Shape from_vector(std::vector<size_t> dimensions) { return Shape{.dimensions = std::move(dimensions)}; }
    static std::vector<size_t> to_vector(Shape shape)        { return std::move(shape.dimensions); }

  public:
    size_t rank() const { return dimensions.size(); }
    size_t dimension(size_t i) const { return dimensions[i]; }

  public:
    size_t size() const
    {
      size_t size = 1;
      for(size_t i=0; i<rank(); ++i)
        size *= dimension(i);
      return size;
    }

  public:
    Shape subshape(size_t begin, size_t count) const
    {
      assert(begin+count<=dimensions.size());

      Shape result;
      for(size_t i=begin; i<begin+count; ++i)
        result.dimensions.push_back(dimensions[i]);
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
    friend bool operator==(const Shape& lhs, const Shape& rhs) = default;
    friend bool operator!=(const Shape& lhs, const Shape& rhs) = default;
  };

  Shape Shape::make(const auto&... froms)
  {
    Shape result;
    auto f = [&result](const auto& from) {
      if constexpr(std::is_convertible_v<std::remove_cvref_t<decltype(from)>, size_t>)
        result.dimensions.push_back(from);
      else if constexpr(std::is_same_v<std::remove_cvref_t<decltype(from)>, Shape>)
        for(size_t dimension : from.dimensions)
          result.dimensions.push_back(dimension);
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
