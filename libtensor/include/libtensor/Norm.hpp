#pragma once

// Declaration
#include <libtensor/Export.hpp>
#include <libtensor/Tensor.hpp>

#include <cmath>

namespace tensor
{
  template<typename T> LIBTENSOR_EXPORT T norm(Tensor<T> a);
}
