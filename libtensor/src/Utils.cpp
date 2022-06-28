#include <libtensor/Utils.hpp>

namespace tensor::utils
{
  size_t max_coeff(Tensor<const float> value)
  {
    value = value.flatten();

    size_t max_coeff = 0;
    float max_value = -std::numeric_limits<float>::infinity();
    for(size_t i=0; i<value.size(); ++i)
      if(value(i) > max_value)
      {
        max_coeff = i;
        max_value = value(i);
      }

    return max_coeff;
  }
}
