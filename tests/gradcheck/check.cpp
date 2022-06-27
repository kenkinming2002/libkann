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

kann::Tensor<float> with_unit_batching(kann::Tensor<float> value)
{
  kann::Shape shape = value.shape();
  return std::move(value).reshape(kann::Shape::concat(kann::Shape(1), shape));
}

kann::Tensor<float> without_unit_batching(kann::Tensor<float> value)
{
  kann::Shape shape = value.shape();

  // No need to check explicitly as the assertion in reshape would catch any size mismatch
  // if the first dimension either does not exist or is not 1
  return std::move(value).reshape(shape.drop_front(1));
}

struct LayerDerivative
{
  kann::Tensor<float> input;
  std::vector<kann::Tensor<float>> parameters;

  static LayerDerivative create_uninitialized_for(const kann::Layer& layer)
  {
    const kann::Shape input_shape  = layer.def->get_input_shape();
    const kann::Shape output_shape = layer.def->get_output_shape();

    LayerDerivative derivative{ .input = kann::Tensor<float>::create(kann::Shape::concat(output_shape, input_shape)) };
    for(const kann::Variable* parameter : layer.storage->get_parameters())
    {
      const kann::Shape parameter_shape = parameter->value.shape();
      derivative.parameters.push_back(kann::Tensor<float>::create(kann::Shape::concat(output_shape, parameter_shape)));
    }
    return derivative;
  }
};

kann::Tensor<float> create_basis(kann::Shape shape, size_t i)
{
  kann::Tensor<float> result = kann::Tensor<float>::create(shape);
  kann::Tensor<float> _result = result.flatten();
  _result.fill(0.0f);
  _result(i) = 1.0f;
  return result;
}

inline LayerDerivative compute_analytical_derivative(kann::Layer& layer, const kann::Tensor<float>& random_input)
{
  const kann::Shape input_shape  = layer.def->get_input_shape();
  const kann::Shape output_shape = layer.def->get_output_shape();

  const size_t input_size  = input_shape.size();
  const size_t output_size = output_shape.size();

  LayerDerivative derivative = LayerDerivative::create_uninitialized_for(layer);
  for(size_t j=0; j<output_size; ++j)
  {
    kann::Tensor<float> input  = with_unit_batching(random_input.clone());
    kann::Tensor<float> output = without_unit_batching(layer.forward(input.clone()));
    (void)output;

    kann::Tensor<float> output_gradient = with_unit_batching(create_basis(output_shape, j));
    kann::Tensor<float> input_gradient  = without_unit_batching(layer.backward(output_gradient.clone()));

    // Derivative for input
    {
      kann::Tensor<float> _input_derivative = derivative.input.reshape(kann::Shape(output_size, input_size));
      kann::Tensor<float> _input_gradient   = input_gradient.reshape(kann::Shape(input_size));
      for(size_t i=0; i<input_size; ++i)
        _input_derivative(j,i) = _input_gradient(i);
    }

    // Derivative for parameters
    {
      const std::vector<kann::Variable*> parameters = layer.storage->get_parameters();
      const size_t parameters_count = parameters.size();
      for(size_t k=0; k<parameters_count; ++k)
      {
        const kann::Shape parameter_shape = parameters[k]->shape;
        const size_t parameter_size = parameter_shape.size();

        kann::Tensor<float> _parameter_derivative = derivative.parameters[k].reshape(kann::Shape(output_size, parameter_size));
        kann::Tensor<float> _parameter_gradient   = parameters[k]->gradient .reshape(kann::Shape(parameter_size));
        for(size_t i=0; i<parameter_size; ++i)
          _parameter_derivative(j,i) = _parameter_gradient(i);
      }
    }
  }

  return derivative;
}

kann::Tensor<float> perturb(kann::Tensor<float> value, size_t i, float diff)
{
  value.flatten()(i) += diff;
  return value;
}

inline LayerDerivative compute_numerical_derivative(kann::Layer& layer, const kann::Tensor<float>& random_input, float dx)
{
  const kann::Shape input_shape  = layer.def->get_input_shape();
  const kann::Shape output_shape = layer.def->get_output_shape();

  const size_t input_size  = input_shape.size();
  const size_t output_size = output_shape.size();

  LayerDerivative derivative = LayerDerivative::create_uninitialized_for(layer);

  // Derivative with respect to inputs
  {
    for(size_t i=0; i<input_size; ++i)
    {
      kann::Tensor<float> input1  = with_unit_batching(random_input.clone());
      kann::Tensor<float> output1 = without_unit_batching(layer.forward(input1.clone()));

      kann::Tensor<float> input2  = with_unit_batching(perturb(random_input.clone(), i, dx));
      kann::Tensor<float> output2 = without_unit_batching(layer.forward(input2.clone()));

      kann::Tensor<float> output_gradient = output2.clone();
      kann::math::transform<1>(output_gradient.flatten(), { output1.flatten() },         kann::math::SUB);
      kann::math::transform<1>(output_gradient.flatten(), { output_gradient.flatten() }, kann::math::SCALE(1.0f/dx));

      kann::Tensor<float> _input_derivative  = derivative.input.reshape(kann::Shape(output_size, input_size));
      kann::Tensor<float> _output_gradient   = output_gradient .reshape(kann::Shape(output_size));
      for(size_t j=0; j<output_size; ++j)
        _input_derivative(j,i) = _output_gradient(j);
    }
  }

  // Derivative with respect to parameters
  {
    const std::vector<kann::Variable*> parameters = layer.storage->get_parameters();
    const size_t parameters_count = parameters.size();
    for(size_t k=0; k<parameters_count; ++k)
    {
      const kann::Shape parameter_shape = parameters[k]->shape;
      const size_t parameter_size = parameter_shape.size();

      // This time we perturb the parameter instead of input
      for(size_t i=0; i<parameter_size; ++i)
      {
        kann::Tensor<float> input1  = with_unit_batching(random_input.clone());
        kann::Tensor<float> output1 = without_unit_batching(layer.forward(input1.clone()));

        // Save and perturb
        kann::Tensor<float> saved_parameter = std::move(parameters[k]->value);
        parameters[k]->value = perturb(saved_parameter.clone(), i, dx);

        kann::Tensor<float> input2  = with_unit_batching(random_input.clone());
        kann::Tensor<float> output2 = without_unit_batching(layer.forward(input2.clone()));

        // Restore
        parameters[k]->value = std::move(saved_parameter);

        kann::Tensor<float> output_gradient = output2.clone();
        kann::math::transform<1>(output_gradient.flatten(), { output1.flatten() },         kann::math::SUB);
        kann::math::transform<1>(output_gradient.flatten(), { output_gradient.flatten() }, kann::math::SCALE(1.0f/dx));

        kann::Tensor<float> _parameter_derivative = derivative.parameters[k].reshape(kann::Shape(output_size, parameter_size));
        kann::Tensor<float> _output_gradient      = output_gradient.reshape(kann::Shape(output_size));
        for(size_t j=0; j<output_size; ++j)
          _parameter_derivative(j,i) = _output_gradient(j);
      }
    }
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
  random_input.fill_uniform(prng, -1.0f, 1.0f);

  LayerDerivative analytical = compute_analytical_derivative(*layer, random_input);
  LayerDerivative numerical  = compute_numerical_derivative(*layer, random_input, DX);

  kann::Tensor<const float> _analytical_derivative = analytical.input.flatten();
  kann::Tensor<const float> _numerical_derivative  = numerical.input.flatten();
  for(size_t i=0; i<_analytical_derivative.size(); ++i)
  {
    const float analytical_value = _analytical_derivative(i);
    const float numerical_value  = _numerical_derivative(i);
    REQUIRE(std::abs(analytical_value - numerical_value) <= 0.01f);
  }

  for(const auto& [analytical_derivative, numerical_derivative] : ranges::views::zip(analytical.parameters, numerical.parameters))
  {
    kann::Tensor<const float> _analytical_derivative = analytical_derivative.flatten();
    kann::Tensor<const float> _numerical_derivative  = numerical_derivative.flatten();
    for(size_t i=0; i<_analytical_derivative.size(); ++i)
    {
      const float analytical_value = _analytical_derivative(i);
      const float numerical_value  = _numerical_derivative(i);
      REQUIRE(std::abs(analytical_value - numerical_value) <= 0.01f);
    }
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
    layer_def->input_size = kann::Vec2(5, 8);
    layer_def->output_size = kann::Vec2(2, 4);
    layer_def->kernel_size = kann::Vec2(4, 5);
    test_layer_def(layer_def);
  }

  SECTION("Dense Layer")
  {
    auto layer_def = std::make_shared<kann::DenseLayerDef>();
    layer_def->input_shape  = kann::Shape{3, 2};
    layer_def->output_shape = kann::Shape{5, 3};
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
