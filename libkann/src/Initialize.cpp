#include <libkann/Initialize.hpp>

#include <libkann/LayerDef.hpp>

#include <libkann/layer_defs/Activation.hpp>
#include <libkann/layer_defs/Convolutional.hpp>
#include <libkann/layer_defs/Sequential.hpp>
#include <libkann/layer_defs/Dense.hpp>
#include <libkann/layer_defs/SoftMax.hpp>

#include <fenv.h>

namespace kann
{
  void initialize()
  {
    feenableexcept(FE_INVALID);

    LayerDef::register_save_load<ActivationLayerDef>("activation");
    LayerDef::register_save_load<ConvolutionalLayerDef>("convolution");
    LayerDef::register_save_load<SequentialLayerDef>("sequential");
    LayerDef::register_save_load<DenseLayerDef>("dense");
    LayerDef::register_save_load<SoftMaxLayerDef>("softmax");
  }
}
