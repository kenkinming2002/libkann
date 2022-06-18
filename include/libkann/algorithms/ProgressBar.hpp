#pragma once

#include <math.h>
#include <string_view>

namespace kann
{
  class ProgressBar
  {
  public:
    ProgressBar(std::string_view title, size_t total_count);

  public:
    void update(std::string_view content, size_t count = 1);

  private:
    std::string_view m_title;
    size_t m_current_count;
    size_t m_total_count;
    size_t m_width;
  };
}
