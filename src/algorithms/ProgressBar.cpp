#include <libkann/algorithms/ProgressBar.hpp>

#include <fmt/core.h>

namespace kann
{
  ProgressBar::ProgressBar(std::string_view title, size_t total_count)
  {
    m_title         = title;
    m_current_count = 0;
    m_total_count   = total_count;
    m_width         = floor(log10(total_count))+1;

    fmt::print("\e[?25l");
  }

  void ProgressBar::update(std::string_view content, size_t count)
  {
    m_current_count += count;
    fmt::print("{} - {:{}}/{} - {}\r", m_title, m_current_count, m_width, m_total_count, content);
    if(m_current_count >= m_total_count)
      fmt::print("\n\e[?25h");
  }
}
