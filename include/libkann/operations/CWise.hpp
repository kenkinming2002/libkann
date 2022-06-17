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
  std::vector<Tensor> operation_process_cwise_impl(std::vector<Tensor> inputs, Shape shape, Impl impl)
  {
    return [&]<std::size_t... Is, std::size_t... Js>(std::index_sequence<Is...>, std::index_sequence<Js...>) {
      assert(inputs.size() == M);
      assert(ranges::all_of(inputs, [&shape](const Tensor& input) { return input.shape() == shape; }));

      std::array<MutableTensor, N> outputs{((void)Js, MutableTensor::create(shape))...};
      const size_t size = shape.size();
      for(size_t i = 0; i<size; ++i)
      {
        auto cwise_outputs = impl(inputs[Is].get(i)...);
        static_assert(std::tuple_size_v<decltype(cwise_outputs)> == N);
        ((outputs[Js].get(i) = std::get<Js>(cwise_outputs)), ...);
      }

      return std::vector{outputs[Js].as_const()...};
    }(std::make_index_sequence<M>(), std::make_index_sequence<N>());
  }
}
