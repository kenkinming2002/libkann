#include <libkann/Initialize.hpp>

#include <libkann/LayerDef.hpp>

#include <libkann/layers/Activation.hpp>
#include <libkann/layers/Convolutional.hpp>
#include <libkann/layers/Sequential.hpp>
#include <libkann/layers/Dense.hpp>
#include <libkann/layers/SoftMax.hpp>

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
