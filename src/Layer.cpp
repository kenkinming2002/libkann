#include <libkann/Layer.hpp>

namespace kann
{
  Tensor Layer::forward(Tensor input)
  {
    return this->def->forward(*this, std::move(input));
  }

  Tensor Layer::backward(Tensor output_gradient)
  {
    return this->def->backward(*this, std::move(output_gradient));
  }
}
