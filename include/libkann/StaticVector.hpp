#pragma once

#include <range/v3/all.hpp>

#include <array>
#include <type_traits>

#include <assert.h>
#include <stddef.h>

namespace kann
{
  template<typename T, size_t Capacity, typename = void>
  struct StaticVector;

  template<typename T, size_t Capacity>
  struct StaticVector<T, Capacity, std::enable_if_t<std::is_trivial_v<T>>>
  {
  public:
    constexpr StaticVector() = default;
    constexpr StaticVector(const StaticVector& other)
    {
      for(const T& value : other)
        push_back(value);
    }

    constexpr StaticVector& operator=(const StaticVector& other)
    {
      clear();
      for(const T& value : other)
        push_back(value);

      return *this;
    }

  public:
    constexpr size_t size() const { return m_size; }
    constexpr void clear() { m_size = 0; }

  public:
    constexpr const T& operator[](size_t i) const
    {
      assert(i<m_size);
      return m_data[i];
    }

    constexpr T& operator[](size_t i)
    {
      assert(i<m_size);
      return m_data[i];
    }

  public:
    constexpr void push_back(const T& value)
    {
      assert(m_size<Capacity);
      m_data[m_size++] = value;
    }

    constexpr void pop_back()
    {
      assert(m_size>0);
      --m_size;
    }

  public:
    constexpr T* begin() { return &m_data[0]; }
    constexpr T* end()   { return &m_data[m_size]; }

    constexpr const T* begin() const { return &m_data[0]; }
    constexpr const T* end() const   { return &m_data[m_size]; }

  private:
    size_t m_size = 0;
    std::array<T, Capacity> m_data;
  };

  template<typename T, size_t Capacity>
  bool operator==(const StaticVector<T, Capacity>& lhs, const StaticVector<T, Capacity>& rhs)
  {
    return ranges::equal(lhs, rhs);
  }

  template<typename T, size_t Capacity>
  bool operator!=(const StaticVector<T, Capacity>& lhs, const StaticVector<T, Capacity>& rhs)
  {
    return lhs != rhs;
  }
}
