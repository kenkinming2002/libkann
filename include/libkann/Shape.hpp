#pragma once

#include <array>
#include <vector>

#include <range/v3/all.hpp>

#include <stddef.h>
#include <assert.h>

namespace kann
{
  struct Shape
  {
  public:
    static constexpr size_t MAX_DIMENSION = 6;

  public:
    static Shape from_vector(std::vector<size_t> dimensions)
    {
      assert(dimensions.size() <= MAX_DIMENSION);
      Shape shape;
      shape.m_dimension_count = dimensions.size();
      for(size_t i=0; i<dimensions.size(); ++i)
        shape.m_dimensions[i] = dimensions[i];
      return shape;
    }

    static std::vector<size_t> to_vector(Shape shape)
    {
      return shape.m_dimensions
        | ranges::views::take_exactly(shape.m_dimension_count)
        | ranges::to_vector;
    }

  public:
    constexpr Shape() : m_dimension_count(0) {}
    template<typename... Dimensions>
    constexpr explicit Shape(Dimensions... dimensions)
      : m_dimension_count(sizeof...(Dimensions)),
        m_dimensions{static_cast<size_t>(dimensions)...} {}

  public:
    constexpr size_t dimension_count() const { return m_dimension_count; }
    constexpr size_t dimension(size_t i) const
    {
      assert(i<dimension_count());
      return m_dimensions[i];
    }

  public:
    constexpr size_t size() const
    {
      size_t size = 1;
      for(size_t i=0; i<dimension_count(); ++i)
        size *= dimension(i);
      return size;
    }

  public:
    bool is_scalar() const { return dimension_count() == 0; }
    bool is_vector() const { return dimension_count() == 1; }
    bool is_matrix() const { return dimension_count() == 2; }

  public:
    constexpr std::pair<Shape, Shape> split(size_t rank1, size_t rank2) const
    {
      assert(rank1+rank2==m_dimension_count);

      Shape result1, result2;
      result1.m_dimension_count = rank1;
      result2.m_dimension_count = rank2;
      for(size_t i=0; i<rank1; ++i)
        result1.m_dimensions[i] = m_dimensions[i];
      for(size_t i=0; i<rank2; ++i)
        result2.m_dimensions[i] = m_dimensions[rank1+i];
      return std::make_pair(result1, result2);
    }

    constexpr std::tuple<Shape, Shape, Shape> split(size_t rank1, size_t rank2, size_t rank3) const
    {
      assert(rank1+rank2==m_dimension_count);

      Shape result1, result2, result3;
      result1.m_dimension_count = rank1;
      result2.m_dimension_count = rank2;
      result3.m_dimension_count = rank3;
      for(size_t i=0; i<rank1; ++i)
        result1.m_dimensions[i] = m_dimensions[i];
      for(size_t i=0; i<rank2; ++i)
        result2.m_dimensions[i] = m_dimensions[rank1+i];
      for(size_t i=0; i<rank3; ++i)
        result3.m_dimensions[i] = m_dimensions[rank1+rank2+i];
      return std::make_tuple(result1, result2, result3);
    }

    static constexpr Shape concat(Shape a, Shape b)
    {
      Shape shape;
      shape.m_dimension_count = a.m_dimension_count + b.m_dimension_count;
      assert(shape.m_dimension_count <= MAX_DIMENSION);
      for(size_t i=0; i<a.m_dimension_count; ++i)
        shape.m_dimensions[i] = a.m_dimensions[i];
      for(size_t i=0; i<b.m_dimension_count; ++i)
        shape.m_dimensions[a.m_dimension_count+i] = b.m_dimensions[i];
      return shape;
    }

    static constexpr Shape concat(Shape a, Shape b, Shape c)
    {
      Shape shape;
      shape.m_dimension_count = a.m_dimension_count + b.m_dimension_count + c.m_dimension_count;
      assert(shape.m_dimension_count <= MAX_DIMENSION);
      for(size_t i=0; i<a.m_dimension_count; ++i)
        shape.m_dimensions[i] = a.m_dimensions[i];
      for(size_t i=0; i<b.m_dimension_count; ++i)
        shape.m_dimensions[a.m_dimension_count+i] = b.m_dimensions[i];
      for(size_t i=0; i<c.m_dimension_count; ++i)
        shape.m_dimensions[a.m_dimension_count+b.m_dimension_count+i] = c.m_dimensions[i];
      return shape;
    }

  public:
    constexpr Shape front(size_t count) const      { assert(m_dimension_count>=count); return this->split(count, m_dimension_count - count).first; }
    constexpr Shape drop_front(size_t count) const { assert(m_dimension_count>=count); return this->split(count, m_dimension_count - count).second; }

    constexpr Shape back(size_t count) const      { assert(m_dimension_count>=count); return this->split(m_dimension_count - count, count).second; }
    constexpr Shape drop_back(size_t count) const { assert(m_dimension_count>=count); return this->split(m_dimension_count - count, count).first; }

  private:
    // TODO: Use fixed size vector
    size_t m_dimension_count;
    std::array<size_t, MAX_DIMENSION> m_dimensions;
  };

  inline bool operator==(const Shape& lhs, const Shape& rhs)
  {
    if(lhs.dimension_count() != rhs.dimension_count())
      return false;

    for(size_t i=0; i<lhs.dimension_count(); ++i)
      if(lhs.dimension(i) != rhs.dimension(i))
        return false;

    return true;
  }

  inline bool operator!=(const Shape& lhs, const Shape& rhs)
  {
    return !(lhs == rhs);
  }

}
