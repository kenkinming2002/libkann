#include <libkann/Initialize.hpp>

#include <libkann/SL.hpp>

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

    layer_def_sl_register<ActivationLayerDef>("activation");
    layer_def_sl_register<ConvolutionalLayerDef>("convolution");
    layer_def_sl_register<SequentialLayerDef>("sequential");
    layer_def_sl_register<DenseLayerDef>("dense");
    layer_def_sl_register<SoftMaxLayerDef>("softmax");
  }
}
