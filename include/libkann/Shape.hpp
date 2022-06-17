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
      shape.m_rank = dimensions.size();
      for(size_t i=0; i<dimensions.size(); ++i)
        shape.m_dimensions[i] = dimensions[i];
      return shape;
    }

    static std::vector<size_t> to_vector(Shape shape)
    {
      return shape.m_dimensions
        | ranges::views::take_exactly(shape.m_rank)
        | ranges::to_vector;
    }

  public:
    constexpr Shape() : m_rank(0) {}
    template<typename... Dimensions>
    constexpr explicit Shape(Dimensions... dimensions)
      : m_rank(sizeof...(Dimensions)),
        m_dimensions{static_cast<size_t>(dimensions)...} {}

  public:
    constexpr size_t rank() const { return m_rank; }
    constexpr size_t dimension(size_t i) const
    {
      assert(i<rank());
      return m_dimensions[i];
    }

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
      assert(rank1+rank2==m_rank);

      Shape result1, result2;
      result1.m_rank = rank1;
      result2.m_rank = rank2;
      for(size_t i=0; i<rank1; ++i)
        result1.m_dimensions[i] = m_dimensions[i];
      for(size_t i=0; i<rank2; ++i)
        result2.m_dimensions[i] = m_dimensions[rank1+i];
      return std::make_pair(result1, result2);
    }

    constexpr std::tuple<Shape, Shape, Shape> split(size_t rank1, size_t rank2, size_t rank3) const
    {
      assert(rank1+rank2==m_rank);

      Shape result1, result2, result3;
      result1.m_rank = rank1;
      result2.m_rank = rank2;
      result3.m_rank = rank3;
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
      shape.m_rank = a.m_rank + b.m_rank;
      assert(shape.m_rank <= MAX_DIMENSION);
      for(size_t i=0; i<a.m_rank; ++i)
        shape.m_dimensions[i] = a.m_dimensions[i];
      for(size_t i=0; i<b.m_rank; ++i)
        shape.m_dimensions[a.m_rank+i] = b.m_dimensions[i];
      return shape;
    }

    static constexpr Shape concat(Shape a, Shape b, Shape c)
    {
      Shape shape;
      shape.m_rank = a.m_rank + b.m_rank + c.m_rank;
      assert(shape.m_rank <= MAX_DIMENSION);
      for(size_t i=0; i<a.m_rank; ++i)
        shape.m_dimensions[i] = a.m_dimensions[i];
      for(size_t i=0; i<b.m_rank; ++i)
        shape.m_dimensions[a.m_rank+i] = b.m_dimensions[i];
      for(size_t i=0; i<c.m_rank; ++i)
        shape.m_dimensions[a.m_rank+b.m_rank+i] = c.m_dimensions[i];
      return shape;
    }

  public:
    constexpr Shape front(size_t count) const      { assert(m_rank>=count); return this->split(count, m_rank - count).first; }
    constexpr Shape drop_front(size_t count) const { assert(m_rank>=count); return this->split(count, m_rank - count).second; }

    constexpr Shape back(size_t count) const      { assert(m_rank>=count); return this->split(m_rank - count, count).second; }
    constexpr Shape drop_back(size_t count) const { assert(m_rank>=count); return this->split(m_rank - count, count).first; }

  private:
    // TODO: Use fixed size vector
    size_t m_rank;
    std::array<size_t, MAX_DIMENSION> m_dimensions;
  };

  inline bool operator==(const Shape& lhs, const Shape& rhs)
  {
    if(lhs.rank() != rhs.rank())
      return false;

    for(size_t i=0; i<lhs.rank(); ++i)
      if(lhs.dimension(i) != rhs.dimension(i))
        return false;

    return true;
  }

  inline bool operator!=(const Shape& lhs, const Shape& rhs)
  {
    return !(lhs == rhs);
  }

}
