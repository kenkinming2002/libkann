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

static kann::Tensor<float> with_unit_batching(kann::Tensor<float> value)
{
  kann::Shape shape = value.as_ref().shape();
  return std::move(value).reshape(kann::Shape::concat(kann::Shape(1), shape));
}

static kann::Tensor<float> without_unit_batching(kann::Tensor<float> value)
{
  kann::Shape shape = value.as_ref().shape();

  // No need to check explicitly as the assertion in reshape would catch any size mismatch
  // if the first dimension either does not exist or is not 1
  return std::move(value).reshape(shape.drop_front(1));
}

struct LayerDerivative
{
  kann::Tensor<float> input;
};

static kann::Tensor<float> create_basis(kann::Shape shape, size_t i)
{
  kann::Tensor<float> result = kann::Tensor<float>::create(shape);
  kann::TensorRef<float> _result = result.as_ref().flatten();
  _result.fill(0.0f);
  _result[i].as_scalar() = 1.0f;
  return result;
}

static inline LayerDerivative compute_analytical_derivative(kann::Layer& layer, const kann::Tensor<float>& random_input)
{
  const kann::Shape input_shape  = layer.def->get_input_shape();
  const kann::Shape output_shape = layer.def->get_output_shape();

  const size_t input_size  = input_shape.size();
  const size_t output_size = output_shape.size();

  LayerDerivative derivative{ .input = kann::Tensor<float>::create(kann::Shape::concat(output_shape, input_shape)) };
  for(size_t j=0; j<output_size; ++j)
  {
    kann::Tensor<float> input  = with_unit_batching(random_input.clone());
    kann::Tensor<float> output = without_unit_batching(layer.forward(input.clone()));
    (void)output;

    kann::Tensor<float> output_gradient = with_unit_batching(create_basis(output_shape, j));
    kann::Tensor<float> input_gradient  = without_unit_batching(layer.backward(output_gradient.clone()));

    kann::TensorRef<float>       _input_derivative = derivative.input.as_ref().reshape(kann::Shape(output_size, input_size));
    kann::TensorRef<const float> _input_gradient   = input_gradient.as_const_ref().reshape(kann::Shape(input_size));
    for(size_t i=0; i<input_size; ++i)
      _input_derivative[j][i].as_scalar() = _input_gradient[i].as_scalar();
  }

  return derivative;
}

static kann::Tensor<float> perturb(kann::Tensor<float> value, size_t i, float diff)
{
  value.as_ref().flatten()[i].as_scalar() += diff;
  return value;
}

static inline LayerDerivative compute_numerical_derivative(kann::Layer& layer, const kann::Tensor<float>& random_input, float dx)
{
  const kann::Shape input_shape  = layer.def->get_input_shape();
  const kann::Shape output_shape = layer.def->get_output_shape();

  const size_t input_size  = input_shape.size();
  const size_t output_size = output_shape.size();

  LayerDerivative derivative{ .input = kann::Tensor<float>::create(kann::Shape::concat(output_shape, input_shape)) };
  for(size_t i=0; i<input_size; ++i)
  {
    kann::Tensor<float> input1  = with_unit_batching(random_input.clone());
    kann::Tensor<float> output1 = without_unit_batching(layer.forward(input1.clone()));

    kann::Tensor<float> input2  = with_unit_batching(perturb(random_input.clone(), i, dx));
    kann::Tensor<float> output2 = without_unit_batching(layer.forward(input2.clone()));

    kann::Tensor<float> output_gradient = output2.clone();
    kann::math::transform<1>(output_gradient.as_ref(), { output1.as_const_ref() },         kann::math::SUB);
    kann::math::transform<1>(output_gradient.as_ref(), { output_gradient.as_const_ref() }, kann::math::SCALE(1.0f/dx));

    kann::TensorRef<float>       _input_derivative  = derivative.input.as_ref().reshape(kann::Shape(output_size, input_size));
    kann::TensorRef<const float> _output_gradient   = output_gradient.as_const_ref().reshape(kann::Shape(output_size));
    for(size_t j=0; j<output_size; ++j)
      _input_derivative[j][i].as_scalar() = _output_gradient[j].as_scalar();
  }
  return derivative;
}

static inline std::shared_ptr<kann::Layer> create_layer(std::shared_ptr<const kann::LayerDef> layer_def, auto& prng)
{
  std::shared_ptr<kann::LayerStorage> layer_storage = layer_def->create(prng);
  return kann::Layer::create_from(layer_def, layer_storage);
}

static inline void test_layer(std::shared_ptr<kann::Layer> layer, auto& prng)
{
  static constexpr float DX = 0.01f;

  kann::Tensor<float> random_input = kann::Tensor<float>::create(layer->def->get_input_shape());
  random_input.as_ref().fill_uniform(prng, -1.0f, 1.0f);

  LayerDerivative analytical = compute_analytical_derivative(*layer, random_input);
  LayerDerivative numerical  = compute_numerical_derivative(*layer, random_input, DX);

  kann::TensorRef<const float> _analytical_derivative = analytical.input.as_const_ref().flatten();
  kann::TensorRef<const float> _numerical_derivative  = numerical.input.as_const_ref().flatten();
  for(size_t i=0; i<_analytical_derivative.size(); ++i)
  {
    const float analytical = _analytical_derivative[i].as_scalar();
    const float numerical  = _numerical_derivative[i].as_scalar();
    REQUIRE(std::abs(analytical - numerical) <= 0.01f);
  }
}

static void test_layer_def(std::shared_ptr<const kann::LayerDef> layer_def)
{
  static std::default_random_engine prng(kann::random<std::default_random_engine::result_type>());

  std::shared_ptr<kann::Layer> layer = create_layer(std::move(layer_def), prng);
  test_layer(std::move(layer), prng);
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
