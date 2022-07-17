#include <libkann/Layer.hpp>

namespace kann
{
  std::shared_ptr<Layer> Layer::from(std::shared_ptr<const LayerDef> def, std::shared_ptr<LayerStorage> storage)
  {
    std::shared_ptr<Layer> layer = std::make_shared<Layer>();
    layer->sub_layers = ranges::views::transform(def->sub_layer_defs, storage->sub_layer_storages, &Layer::from) | ranges::to_vector;
    layer->def        = std::move(def);
    layer->storage    = std::move(storage);
    return layer;
  }

  std::shared_ptr<Layer> Layer::load_and_create_from(const std::string& filename, std::default_random_engine& prng)
  {
    auto def     = LayerDef::load(filename);
    auto storage = def->create(prng);
    return from(std::move(def), std::move(storage));
  }

  tensor::Tensor<float> Layer::forward(tensor::Tensor<float> inputs)
  {
    return this->def->forward(*this, std::move(inputs));
  }

  tensor::Tensor<float> Layer::backward(tensor::Tensor<float> output_gradients)
  {
    return this->def->backward(*this, std::move(output_gradients));
  }
}
