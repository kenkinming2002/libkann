#pragma once

#include <libkann/Types.hpp>
#include <libkann/Tensor.hpp>

#include <vector>
#include <tuple>
#include <type_traits>

namespace kann
{
  /* impl takes multiple argument and return a tuple like type */
  template<size_t M, size_t N, typename Impl>
  std::vector<Tensor> operation_process_impl(std::vector<Tensor> inputs, Impl impl)
  {
    return [&]<std::size_t... Is, std::size_t... Js>(std::index_sequence<Is...>, std::index_sequence<Js...>) {
      assert(inputs.size() == M);
      auto outputs = impl(inputs[Is]...);
      static_assert(std::tuple_size_v<decltype(outputs)> == N);
      return std::vector{std::move(std::get<Js>(outputs))...};
    }(std::make_index_sequence<M>(), std::make_index_sequence<N>());
  }
}
