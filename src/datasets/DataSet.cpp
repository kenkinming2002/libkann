#include <libkann/datasets/DataSet.hpp>

namespace kann
{
  std::vector<std::shared_ptr<const Tensor>> load(const DataSet& data_set, size_t column)
  {
    size_t size = data_set.size();

    std::vector<std::shared_ptr<const Tensor>> data;
    data.reserve(size);
    for(size_t i=0; i<size; ++i)
      data.push_back(data_set.get(column, i));

    return data;
  }
}
