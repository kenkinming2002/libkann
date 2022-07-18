#pragma once

#include <memory>
#include <span>

namespace tensor
{
  template<typename T>
  struct Buffer
  {
  public:
    Buffer() = default;
    Buffer(size_t size)
    {
      m_data = std::make_unique_for_overwrite<T[]>(size);
      m_size = size;
    }

  public:
    std::span<const T> data() const { return std::span(m_data.get(), m_size); }
    std::span<T>       data()       { return std::span(m_data.get(), m_size); }

  public:
    const T& operator[](size_t i) const { return m_data[i]; }
    T&       operator[](size_t i)       { return m_data[i]; }

  private:
    std::unique_ptr<T[]> m_data = nullptr;
    size_t               m_size = 0;
  };
}
