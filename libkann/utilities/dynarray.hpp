#pragma once

#include <cstddef>
#include <iterator>
#include <memory>

/* 
 * Partial implementation of dynarray, which has been proposed for c++14,
 * subsequently moved to Array TS, and discontinued now.
 *
 * There is one major difference from the array as described in the proposal in
 * that we do not have the luxury of using the stack memory when the allocation
 * is small and have to rely on heap every time since doing so will require
 * compiler cooperation or we would invoke undefined behavior.
 * 
 * The advantage of dynarray over std::vector is that only a pointer to storage
 * and size of the array is stored whereas in the case of std::vector, there has
 * to be an additional capacity variable. This reduce the size of the
 * container(on the stack) from 24 bytes(most likely) to 16 bytes.
 *
 * The primary advantage of the size reduction will be reduction in memory
 * consumption as std::vector is used a lot throughout this code base. For
 * example, each NeuralNetwork class consists of 3 instances of std::vector. As
 * of now, its size is currently 96 bytes and the usage of dynarray will reduce
 * it by 24 bytes which is a 25% reduction.
 *
 * In addition, we may benefit from better cache locality though the difference
 * may eb negligible.
 *
 * Note that different from std::vector the dynarray container is not allocator
 * aware as 1: YAGNI, 2: I do not know how to correctly implement allocator
 * aware container
 */

template<typename T>
class dynarray
{
public:
  using value_type             = T;
  using reference              = T&;
  using const_reference        = const T&;
  using pointer                = T*;
  using const_pointer          = const T*;
  using iterator               = T*;
  using const_iterator         = const T*;
  using reverse_iterator       = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using difference_type        = std::ptrdiff_t;
  using size_type              = std::size_t;

public:
  dynarray(size_type n) : m_size(n), m_data(std::make_unique<T[]>(n)) {}

  template<typename InputIterator>
  dynarray(InputIterator first, InputIterator last) : m_size(std::distance(first, last)), m_data(std::make_unique<T[]>(m_size)) {}

public:
  dynarray(const dynarray& other) : m_size(other.m_size), m_data(std::make_unique<T[]>(m_size))
  {
    std::copy(other.begin(), other.end(), this->begin());
  }
  dynarray& operator=(const dynarray& other) 
  {
    m_size = other.m_size;
    m_data = std::make_unique<T[]>(m_size);
    std::copy(other.begin(), other.end(), this->begin());
    return *this;
  }

public:
  iterator begin() { return &m_data[0]; }
  iterator end() { return &m_data[m_size]; }

  const_iterator begin() const { return &m_data[0]; }
  const_iterator end() const { return &m_data[m_size]; }

  const_iterator cbegin() const { return &m_data[0]; }
  const_iterator cend() const { return &m_data[m_size]; }

  reverse_iterator rbegin() { return reverse_iterator(end()); }
  reverse_iterator rend() { return reverse_iterator(begin()); }

  const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
  const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

  const_reverse_iterator crbegin() const { return const_reverse_iterator(end()); }
  const_reverse_iterator crend() const { return const_reverse_iterator(begin()); }

public:
  reference operator[](size_type n) { return m_data[n]; }
  const_reference operator[](size_type n) const { return m_data[n]; }

public:
  reference front() { return m_data[0]; }
  const_reference front() const { return m_data[0]; }

public:
  reference back() { return m_data[m_size-1]; }
  const_reference back() const { return m_data[m_size-1]; }

public:
  size_type size() const { return m_size; }

private:
  size_type m_size;
  std::unique_ptr<T[]> m_data;
};
