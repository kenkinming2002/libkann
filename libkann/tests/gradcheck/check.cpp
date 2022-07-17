#include <catch2/catch_all.hpp>
#include <catch2/catch_session.hpp>

#include <libkann/Initialize.hpp>

#include <libtensor/Stack.hpp>
#include <libtensor/Map.hpp>

#include <libkann/Layer.hpp>
#include <libkann/LayerDef.hpp>
#include <libkann/LayerStorage.hpp>

#include <libkann/layer_defs/Activation.hpp>
#include <libkann/layer_defs/Convolutional.hpp>
#include <libkann/layer_defs/Dense.hpp>
#include <libkann/layer_defs/SoftMax.hpp>

using Tensor = tensor::Tensor<const float>;

struct LayerDerivative
{
  Tensor input;
  std::vector<Tensor> parameters;
};

tensor::Tensor<float> create_basis(size_t size, size_t i)
{
  tensor::Tensor<float> result = tensor::Tensor<float>::create(tensor::Shape::make(size));
  std::fill_n(result.data(), result.size(), 0.0f);
  result.data()[i] = 1.0f;
  return result;
}

inline LayerDerivative compute_analytical_derivative(kann::Layer& layer, const Tensor& random_input)
{
  const auto input_shape  = layer.def->get_input_shape();
  const auto output_shape = layer.def->get_output_shape();

  auto parameters = layer.storage->get_parameters();
  const size_t parameters_count = parameters.size();

  LayerDerivative derivative;
  derivative.parameters.resize(parameters_count);

  std::vector<Tensor> input_gradients;
  std::vector<std::vector<Tensor>> parameters_gradients;
  parameters_gradients.resize(parameters_count);

  for(size_t j=0; j<output_shape.size(); ++j)
  {
    auto input           = random_input.reshape(tensor::Shape::make(1, input_shape));
    auto output          = layer.forward(std::move(input)).flatten();

    auto output_gradient = create_basis(output_shape.size(), j).reshape(tensor::Shape::make(1, output_shape));
    auto input_gradient  = layer.backward(std::move(output_gradient)).flatten();

    input_gradients.push_back(std::move(input_gradient));
    parameters_gradients.push_back({});
    for(size_t k=0; k<parameters_count; ++k)
      parameters_gradients[k].push_back(parameters[k]->gradient);
  }

  derivative.input = tensor::stack_outer(input_gradients);
  for(size_t k=0; k<parameters_count; ++k)
    derivative.parameters[k] = tensor::stack_outer(parameters_gradients[k]);
  return derivative;
}

Tensor perturb(Tensor value, size_t i, float diff)
{
  auto result = value.clone();
  result.flatten()(i) += diff;
  return result;
}

inline LayerDerivative compute_numerical_derivative(kann::Layer& layer, const Tensor& random_input, float dx)
{
  const tensor::Shape input_shape  = layer.def->get_input_shape();
  const tensor::Shape output_shape = layer.def->get_output_shape();

  LayerDerivative derivative;

  // Derivative with respect to inputs
  std::vector<Tensor> output_gradients;
  for(size_t i=0; i<input_shape.size(); ++i)
  {
    auto input1  = random_input.reshape(tensor::Shape::make(1, input_shape));
    auto output1 = layer.forward(input1).flatten();

    auto input2  = perturb(random_input, i, dx).reshape(tensor::Shape::make(1, input_shape));
    auto output2 = layer.forward(input2).flatten();

    auto output_gradient = tensor::binary_map(output2, output1, [&dx](float output2, float output1){ return (output2 - output1) / dx; });
    output_gradients.push_back(std::move(output_gradient));
  }
  derivative.input = tensor::stack_inner(output_gradients);

  for(auto* parameter : layer.storage->get_parameters())
  {
    std::vector<Tensor> parameter_gradients;
    for(size_t i=0; i<parameter->shape.size(); ++i)
    {
      auto input   = random_input.reshape(tensor::Shape::make(1, input_shape));

      auto output1 = layer.forward(input).flatten();

      // Save
      auto saved_parameter = std::move(parameter->value);
      parameter->value = perturb(saved_parameter.clone(), i, dx);

      auto output2 = layer.forward(input).flatten();

      // Restore
      parameter->value = saved_parameter;

      auto output_gradient = tensor::binary_map(output2, output1, [&dx](float output2, float output1){ return (output2 - output1) / dx; });
      parameter_gradients.push_back(std::move(output_gradient));
    }
    derivative.parameters.push_back(tensor::stack_inner(parameter_gradients));
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
  static constexpr float DX = 0.0005f;

  auto random_input = tensor::create_uniform(layer->def->get_input_shape(), -1.0f, 1.0f, prng);

  LayerDerivative analytical = compute_analytical_derivative(*layer, random_input);
  LayerDerivative numerical  = compute_numerical_derivative(*layer, random_input, DX);

  Tensor _analytical_derivative = analytical.input.flatten();
  Tensor _numerical_derivative  = numerical.input.flatten();
  for(size_t i=0; i<_analytical_derivative.size(); ++i)
  {
    const float analytical_value = _analytical_derivative(i);
    const float numerical_value  = _numerical_derivative(i);
    REQUIRE(std::abs(analytical_value - numerical_value) <= 0.001f);
  }

  for(const auto& [analytical_derivative, numerical_derivative] : ranges::views::zip(analytical.parameters, numerical.parameters))
  {
    Tensor _analytical_derivative = analytical_derivative.flatten();
    Tensor _numerical_derivative  = numerical_derivative.flatten();
    for(size_t i=0; i<_analytical_derivative.size(); ++i)
    {
      const float analytical_value = _analytical_derivative(i);
      const float numerical_value  = _numerical_derivative(i);
      REQUIRE(std::abs(analytical_value - numerical_value) <= 0.001f);
    }
  }
}

static void test_layer_def(std::shared_ptr<const kann::LayerDef> layer_def)
{
  static std::random_device rd;
  static std::default_random_engine prng(rd());

  std::shared_ptr<kann::Layer> layer = create_layer(std::move(layer_def), prng);
  test_layer(std::move(layer), prng);
}

TEST_CASE("Gradcheck", "[gradcheck]")
{
  SECTION("Activation Layer")
  {
    auto layer_def = std::make_shared<kann::ActivationLayerDef>();
    layer_def->shape = tensor::Shape::make(6,3,8);

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
    layer_def->input_size = tensor::Vec2(5, 8);
    layer_def->output_size = tensor::Vec2(2, 4);
    layer_def->kernel_size = tensor::Vec2(4, 5);
    test_layer_def(layer_def);
  }

  SECTION("Dense Layer")
  {
    auto layer_def = std::make_shared<kann::DenseLayerDef>();
    layer_def->input_shape  = tensor::Shape::make(3, 2);
    layer_def->output_shape = tensor::Shape::make(5, 3);
    test_layer_def(layer_def);
  }

  SECTION("Softmax")
  {
    auto layer_def = std::make_shared<kann::SoftMaxLayerDef>();
    layer_def->shape = tensor::Shape::make(7,3,6);
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
