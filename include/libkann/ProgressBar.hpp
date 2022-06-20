#pragma once

#include <libkann/Export.hpp>

#include <math.h>
#include <string_view>

namespace kann
{
  class ProgressBar
  {
  public:
    KANN_EXPORT ProgressBar(std::string_view title, size_t total_count);

  public:
    KANN_EXPORT void update(std::string_view content, size_t count = 1);

  private:
    std::string_view m_title;
    size_t m_current_count;
    size_t m_total_count;
    size_t m_width;
  };
}
