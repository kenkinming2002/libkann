#include <libkann/operations/IdentityOperation.hpp>

namespace kann
{
  Tensor IdentityOperation::processImpl(const Tensor& input) const
  {
    return input;
  }
}
