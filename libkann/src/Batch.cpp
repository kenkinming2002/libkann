#include <libkann/Batch.hpp>

#include <libtensor/memops/Stack.hpp>
#include <libtensor/memops/Split.hpp>

#include <range/v3/all.hpp>
#include <range/v3/view/join.hpp>

namespace kann
{
  std::vector<tensor::Tensor<float>> batch(std::vector<tensor::Tensor<float>> values, size_t batch_size)
  {
    std::vector<tensor::Tensor<float>> results;
    for(size_t i=0; i+batch_size<=values.size(); i+=batch_size)
      results.push_back(tensor::stack_outer(std::vector(&values[i], &values[i+batch_size])));

    return results;
  }

  std::vector<tensor::Tensor<float>> unbatch(std::vector<tensor::Tensor<float>> values, size_t batch_size)
  {
    std::vector<tensor::Tensor<float>> results;
    for(auto&& value : values)
    {
      auto sub_results = tensor::split_outer(std::move(value));
      assert(sub_results.size() == batch_size);
      results.insert(results.end(), sub_results.begin(), sub_results.end());
    }
    return results;
  }
}

