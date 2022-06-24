#include <catch2/catch_all.hpp>
#include <catch2/catch_session.hpp>

#include <libkann/Initialize.hpp>
#include <libkann/Random.hpp>
#include <libkann/Math.hpp>

#include <libkann/Layer.hpp>
#include <libkann/LayerDef.hpp>
#include <libkann/LayerStorage.hpp>

#include <libkann/layer_defs/Activation.hpp>
#include <libkann/layer_defs/Convolutional.hpp>
#include <libkann/layer_defs/Dense.hpp>
#include <libkann/layer_defs/SoftMax.hpp>

#include <fmt/core.h>

#include <type_traits>

static kann::Tensor<float> create_basis(size_t i, size_t size)
{
  kann::Tensor<float> result = kann::Tensor<float>::create(kann::Shape(size));
  kann::TensorRef<float> _result = result.as_ref();
  _result.fill(0.0f);
  _result[i].as_scalar() = 1.0f;
  return result;
}

static kann::Tensor<float> perturb(const kann::Tensor<float>& value, size_t i, float diff)
{
  kann::Tensor<float> result = value.clone();
  result.as_ref().flatten()[i].as_scalar() += diff;
  return result;
}

static void test_layer_def(std::shared_ptr<const kann::LayerDef> layer_def)
{
  // 1: Setup
  static std::default_random_engine prng(kann::random<std::default_random_engine::result_type>());

  std::shared_ptr<kann::LayerStorage> layer_storage = layer_def->create(prng);
  std::shared_ptr<kann::Layer>        layer         = kann::Layer::create_from(layer_def, layer_storage);

  const kann::Shape input_shape  = layer_def->get_input_shape();
  const kann::Shape output_shape = layer_def->get_output_shape();

  const size_t input_size  = input_shape.size();
  const size_t output_size = output_shape.size();

  // 2: Create random input
  kann::Tensor<float> random_input = kann::Tensor<float>::create(kann::Shape::concat(kann::Shape(1), input_shape));
  random_input.as_ref().fill_uniform(prng, -1.0f, 1.0f);

  // 3: Compute derivatives analytically
  //    To so, we set output_gradient as unit basis tensor
  kann::Tensor<float> analytical_derivative = kann::Tensor<float>::create(kann::Shape::concat(output_shape, input_shape));
  {
    for(size_t j=0; j<output_size; ++j)
    {
      kann::Tensor<float> input  = random_input.clone();
      kann::Tensor<float> output = layer->forward(input.clone());

      // Gradient
      kann::Tensor<float> output_gradient = create_basis(j, output_size);
      output_gradient = std::move(output_gradient).reshape(kann::Shape::concat(kann::Shape(1), output_shape));

      kann::Tensor<float> input_gradient = layer->backward(output_gradient.clone());
      input_gradient = std::move(input_gradient).reshape(kann::Shape(input_size));

      kann::TensorRef<float>       _analytical_derivative = analytical_derivative.as_ref().reshape(kann::Shape(output_size, input_size));
      kann::TensorRef<const float> _input_gradient        = input_gradient.as_const_ref().reshape(kann::Shape(input_size));
      for(size_t i=0; i<input_size; ++i)
        _analytical_derivative[j][i].as_scalar() = _input_gradient[i].as_scalar();
    }
  }

  // 4: Compute gradients numerically
  //    To do so, we set perturb input by basis vector of small length
  kann::Tensor<float> numerical_derivative = kann::Tensor<float>::create(kann::Shape::concat(output_shape, input_shape));
  {
    for(size_t i=0; i<input_size; ++i)
    {
      // We cannot use too small of a DX, as that would introduce inaccurracy
      static constexpr float DX = 0.01f;
      kann::Tensor<float> input1  = random_input.clone();
      kann::Tensor<float> output1 = layer->forward(input1.clone());

      kann::Tensor<float> input2  = perturb(random_input, i, DX);
      kann::Tensor<float> output2 = layer->forward(input2.clone());

      kann::Tensor<float> output_gradient = output2.clone();
      kann::math::transform<1>(output_gradient.as_ref(), { output1.as_const_ref() },         kann::math::SUB);
      kann::math::transform<1>(output_gradient.as_ref(), { output_gradient.as_const_ref() }, kann::math::SCALE(1.0f/DX));

      kann::TensorRef<float>       _numerical_derivative  = numerical_derivative.as_ref().reshape(kann::Shape(output_size, input_size));
      kann::TensorRef<const float> _output_gradient       = output_gradient.as_const_ref().reshape(kann::Shape(output_size));
      for(size_t j=0; j<output_size; ++j)
        _numerical_derivative[j][i].as_scalar() = _output_gradient[j].as_scalar();
    }
  }

  // 5: Compare them
  kann::TensorRef<const float> _analytical_derivative = analytical_derivative.as_const_ref().flatten();
  kann::TensorRef<const float> _numerical_derivative  = numerical_derivative.as_const_ref().flatten();
  for(size_t i=0; i<input_size * output_size; ++i)
  {
    const float analytical = _analytical_derivative[i].as_scalar();
    const float numerical  = _numerical_derivative[i].as_scalar();
    REQUIRE(std::abs(analytical - numerical) <= 0.01f);
  }
}

TEST_CASE("Gradcheck", "[gradcheck]")
{
  SECTION("Activation Layer")
  {
    auto layer_def = std::make_shared<kann::ActivationLayerDef>();
    layer_def->shape = kann::Shape(6,3,8);

    SECTION("Identity")
    {
      layer_def->type = kann::ActivationLayerDef::Type::IDENTITY;
      test_layer_def(layer_def);
    }

    SECTION("Sigmoid")
    {
      layer_def->type = kann::ActivationLayerDef::Type::SIGMOID;
      test_layer_def(layer_def);
    }

    SECTION("Tanh")
    {
      layer_def->type = kann::ActivationLayerDef::Type::TANH;
      test_layer_def(layer_def);
    }
  }

  SECTION("Convolutional Layer")
  {
    auto layer_def = std::make_shared<kann::ConvolutionalLayerDef>();
    layer_def->input_channel_count = 3;
    layer_def->output_channel_count = 4;
    layer_def->input_size = kann::Vec2(20, 20);
    layer_def->output_size = kann::Vec2(15, 15);
    layer_def->kernel_size = kann::Vec2(6, 6);
    test_layer_def(layer_def);
  }

  SECTION("Dense Layer")
  {
    auto layer_def = std::make_shared<kann::DenseLayerDef>();
    layer_def->input_shape  = kann::Shape{11, 23};
    layer_def->output_shape = kann::Shape{47, 21};
    test_layer_def(layer_def);
  }

  SECTION("Softmax")
  {
    auto layer_def = std::make_shared<kann::SoftMaxLayerDef>();
    layer_def->shape = kann::Shape(7,3,6);
    test_layer_def(layer_def);
  }
}

int main(int argc, char* argv[])
{
  kann::initialize();

  Catch::Session session;
  session.applyCommandLine(argc, argv);
  return session.run();
}
