#include <libkann/Initialize.hpp>

#include <libkann/LayerDef.hpp>

#include <libkann/layer_defs/Activation.hpp>
#include <libkann/layer_defs/Convolutional.hpp>
#include <libkann/layer_defs/Sequential.hpp>
#include <libkann/layer_defs/Dense.hpp>

namespace kann
{
  void initialize()
  {
    LayerDef::register_save_load<ActivationLayerDef>("activation");
    LayerDef::register_save_load<ConvolutionalLayerDef>("convolution");
    LayerDef::register_save_load<SequentialLayerDef>("sequential");
    LayerDef::register_save_load<DenseLayerDef>("dense");
  }
}
