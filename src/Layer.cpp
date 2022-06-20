#include <libkann/Layer.hpp>

namespace kann
{
  Tensor Layer::forward(Tensor inputs)
  {
    return this->def->forward(*this, std::move(inputs));
  }

  Tensor Layer::backward(Tensor output_gradients)
  {
    return this->def->backward(*this, std::move(output_gradients));
  }
}
