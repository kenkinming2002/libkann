#include <libkann/Layer.hpp>

namespace kann
{
  std::shared_ptr<Layer> Layer::create_from(std::shared_ptr<const LayerDef> def, std::shared_ptr<LayerStorage> storage)
  {
    std::shared_ptr<Layer> layer = std::make_shared<Layer>();
    layer->def     = std::move(def);
    layer->storage = std::move(storage);
    layer->sub_layers = ranges::views::transform(def->sub_layer_defs, storage->sub_layer_storages, &Layer::create_from) | ranges::to_vector;
    return layer;
  }

  Tensor Layer::forward(Tensor inputs)
  {
    return this->def->forward(*this, std::move(inputs));
  }

  Tensor Layer::backward(Tensor output_gradients)
  {
    return this->def->backward(*this, std::move(output_gradients));
  }
}
