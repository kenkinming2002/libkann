#pragma once

#include <range/v3/view/view.hpp>
#include <range/v3/algorithm/copy.hpp>

#include <array>

namespace kann
{
  template<typename T, size_t N>
  static inline auto to_array()
  {
    return ranges::make_view_closure([](auto&& r){
      std::array<T, N> result;
      assert(ranges::size(std::forward<decltype(r)>(r)) == N);
      ranges::copy(std::forward<decltype(r)>(r), result.begin());
      return result;
    });
  }

}
