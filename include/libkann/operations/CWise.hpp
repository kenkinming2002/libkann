#pragma once

#include <libkann/Types.hpp>
#include <libkann/Tensor.hpp>
#include <libkann/operations/Impl.hpp>

#include <vector>
#include <tuple>
#include <type_traits>

namespace kann
{
  /* impl takes multiple argument and return a tuple like type */
  template<size_t M, size_t N, typename Impl>
  std::vector<tensor_t> operation_process_cwise_impl(std::vector<tensor_t> inputs, size_t size, Impl impl)
  {
    return operation_process_impl<M, N>(std::move(inputs), [&](const auto&... inputs)
    {
      return [&]<std::size_t... Js>(std::index_sequence<Js...>){
        std::array<Tensor, N> outputs;
        for(Tensor& output : outputs)
          output = Tensor(size);

        for(size_t i=0; i<size; ++i)
        {
          auto cwise_outputs = impl(inputs[i]...);
          ((outputs[Js][i] = std::move(std::get<Js>(cwise_outputs))), ...);
        }

        return outputs;
      }(std::make_index_sequence<N>());
    });
  }
}
