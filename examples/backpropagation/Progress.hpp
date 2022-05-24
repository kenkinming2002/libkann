#pragma once

#include <fmt/core.h>

#include <math.h>
#include <string_view>

inline bool sigint_caught = false;
inline void sigint_handler(int)
{
  sigint_caught = true;
}

class ProgressBar
{
public:
  ProgressBar(std::string_view title, size_t total_count)
  {
    m_title         = title;
    m_current_count = 0;
    m_total_count   = total_count;
    m_width         = floor(log10(total_count))+1;

    fmt::print("\e[?25l");
  }

  void update(std::string_view content)
  {
    ++m_current_count;
    fmt::print("{} - {:{}}/{} - {}\r", m_title, m_current_count, m_width, m_total_count, content);
    if(m_current_count == m_total_count)
      fmt::print("\n\e[?25h");
  }

private:
  std::string_view m_title;
  size_t m_current_count;
  size_t m_total_count;
  size_t m_width;
};
