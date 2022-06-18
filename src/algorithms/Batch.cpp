#include "Batch.hpp"

#include <range/v3/all.hpp>
#include <range/v3/view/join.hpp>

namespace kann
{
  std::vector<Tensor> batch(const std::vector<Tensor>& values, size_t batch_size)
  {
    const Shape shape = Shape::concat(Shape(batch_size), values.front().shape());
    if(batch_size != 1)
    {
      return values
        | ranges::views::take_exactly(values.size() / batch_size * batch_size) // Ensure that the each chunk is full
        | ranges::views::chunk(batch_size)
        | ranges::views::transform([&](auto&& sub_values){
            MutableTensor result = MutableTensor::create(shape);
            for(auto&& [i, value] : ranges::views::enumerate(std::forward<decltype(sub_values)>(sub_values)))
              std::copy_n(value.data(), value.size(), result.as_ref()[i].data());

            return result.as_const();
          })
        | ranges::to_vector;
    }
    else
    {
      return values
        | ranges::views::transform([&](const Tensor& value){ return value.reshape(shape); })
        | ranges::to_vector;
    }
  }

  std::vector<Tensor> unbatch(const std::vector<Tensor>& values, size_t batch_size)
  {
    const Shape shape = values.front().shape().drop_front(1);
    if(batch_size != 1)
    {
      return values
        | ranges::views::transform([&](const Tensor& value) {
            return ranges::views::ints(0zu, batch_size)
              | ranges::views::transform([&](size_t i) { return value[i]; })
              | ranges::to_vector;
          })
        | ranges::views::cache1
        | ranges::views::join
        | ranges::to_vector;
    }
    else
    {
      return values
        | ranges::views::transform([&](const Tensor& value){ return value.reshape(shape); })
        | ranges::to_vector;
    }
  }
}

