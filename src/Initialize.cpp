#include <libkann/Initialize.hpp>

#include <libkann/LayerDef.hpp>

#include <libkann/layer_defs/Activation.hpp>
#include <libkann/layer_defs/Convolutional.hpp>
#include <libkann/layer_defs/Deconvolutional.hpp>
#include <libkann/layer_defs/Identity.hpp>
#include <libkann/layer_defs/Recurrent.hpp>
#include <libkann/layer_defs/Sequential.hpp>
#include <libkann/layer_defs/Weight.hpp>

namespace kann
{
  void initialize()
  {
    LayerDef::register_save_load<ActivationLayerDef>("activation");
    LayerDef::register_save_load<ConvolutionalLayerDef>("convolution");
    LayerDef::register_save_load<DeconvolutionalLayerDef>("deconvolution");
    LayerDef::register_save_load<IdentityLayerDef>("identity");
    LayerDef::register_save_load<RecurrentLayerDef>("recurrent");
    LayerDef::register_save_load<SequentialLayerDef>("sequential");
    LayerDef::register_save_load<WeightLayerDef>("weight");
  }
}
