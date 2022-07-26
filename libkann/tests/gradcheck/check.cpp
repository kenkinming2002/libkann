#include <catch2/catch_all.hpp>
#include <catch2/catch_session.hpp>

#include <libkann/Initialize.hpp>

#include <libtensor/memops/Stack.hpp>
#include <libtensor/Map.hpp>

#include <libkann/Layer.hpp>
#include <libkann/LayerDef.hpp>

#include <libkann/layers/Activation.hpp>
#include <libkann/layers/Convolutional.hpp>
#include <libkann/layers/Dense.hpp>
#include <libkann/layers/SoftMax.hpp>

using Buffer = tensor::Buffer<float>;
using Tensor = tensor::Tensor<float>;
using tensor::Shape;

struct LayerDerivative
{
  Tensor input;
  std::unordered_map<std::string, Tensor> parameters;
};

Tensor create_basis(size_t size, size_t i)
{
  auto result_buffer = std::make_shared<Buffer>(size);
  std::fill_n(result_buffer->data().data(), result_buffer->data().size(), 0.0f);
  result_buffer->data()[i] = 1.0f;
  return Tensor(Shape::make(size), std::move(result_buffer));
}

inline LayerDerivative compute_analytical_derivative(kann::Layer& layer, tensor::Shape input_shape, tensor::Shape output_shape, const Tensor& random_input)
{
  std::vector<Tensor> input_gradients;
  std::unordered_map<std::string, std::vector<Tensor>> parameters_gradients;
  for(size_t j=0; j<output_shape.size(); ++j)
  {
    auto input           = random_input.reshape(tensor::Shape::make(1, input_shape));
    auto output          = layer.forward(std::move(input)).flatten();

    auto output_gradient = create_basis(output_shape.size(), j).reshape(tensor::Shape::make(1, output_shape));
    auto input_gradient  = layer.backward(std::move(output_gradient)).flatten();

    input_gradients.push_back(std::move(input_gradient));
    for(const auto& [name, parameter] : layer.parameters_map())
      parameters_gradients[name].push_back(parameter->gradient);
  }

  LayerDerivative derivative;
  derivative.input = tensor::stack_outer(input_gradients);
  for(const auto& [name, parameter] : layer.parameters_map())
    derivative.parameters.insert({name, tensor::stack_outer(parameters_gradients.at(name))});

  return derivative;
}

Tensor perturb(Tensor value, size_t i, float diff)
{
  auto value_buffer  = std::move(value.buffer);
  auto result_buffer = std::make_shared<Buffer>(value_buffer->data().size());
  std::copy_n(value_buffer->data().data(), value_buffer->data().size(), result_buffer->data().data());
  result_buffer->data()[i] += diff;
  return Tensor(std::move(value.shape), std::move(result_buffer));
}

inline LayerDerivative compute_numerical_derivative(kann::Layer& layer, tensor::Shape input_shape, tensor::Shape output_shape, const Tensor& random_input, float dx)
{
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

  for(const auto& [name, parameter] : layer.parameters_map())
  {
    std::vector<Tensor> parameter_gradients;
    for(size_t i=0; i<parameter->shape.size(); ++i)
    {
      auto input   = random_input.reshape(tensor::Shape::make(1, input_shape));

      auto output1 = layer.forward(input).flatten();

      // Save
      auto saved_parameter = std::move(parameter->value);
      parameter->value = perturb(saved_parameter, i, dx);

      auto output2 = layer.forward(input).flatten();

      // Restore
      parameter->value = saved_parameter;

      auto output_gradient = tensor::binary_map(output2, output1, [&dx](float output2, float output1){ return (output2 - output1) / dx; });
      parameter_gradients.push_back(std::move(output_gradient));
    }
    derivative.parameters.insert({name, tensor::stack_inner(parameter_gradients)});
  }
  return derivative;
}

static inline void test_layer(std::shared_ptr<kann::Layer> layer, tensor::Shape input_shape, tensor::Shape output_shape, auto& prng)
{
  static constexpr float DX = 0.0005f;

  auto random_input = tensor::create_uniform(input_shape, -1.0f, 1.0f, prng);

  LayerDerivative analytical = compute_analytical_derivative(*layer, input_shape, output_shape, random_input);
  LayerDerivative numerical  = compute_numerical_derivative(*layer, input_shape, output_shape, random_input, DX);

  auto analytical_values = analytical.input.buffer->data();
  auto numerical_values  = numerical .input.buffer->data();
  for(size_t i=0; i<analytical_values.size(); ++i)
  {
    const float analytical_value = analytical_values[i];
    const float numerical_value  = numerical_values[i];
    REQUIRE(std::abs(analytical_value - numerical_value) <= 0.001f);
  }

  for(const auto& [name, analytical_derivative] : analytical.parameters)
  {
    const auto& numerical_derivative = numerical.parameters.at(name);

    auto analytical_values = analytical_derivative.buffer->data();
    auto numerical_values  = numerical_derivative.buffer->data();

    Tensor _analytical_derivative = analytical_derivative.flatten();
    Tensor _numerical_derivative  = numerical_derivative.flatten();
    for(size_t i=0; i<analytical_values.size(); ++i)
    {
      const float analytical_value = analytical_values[i];
      const float numerical_value  = numerical_values[i];
      REQUIRE(std::abs(analytical_value - numerical_value) <= 0.001f);
    }
  }
}

static void test_layer_def(std::shared_ptr<const kann::LayerDef> layer_def, tensor::Shape input_shape, tensor::Shape output_shape)
{
  static std::random_device rd;
  static std::default_random_engine prng(rd());

  std::shared_ptr<kann::Layer> layer = layer_def->create();
  layer->initialize(prng);
  test_layer(std::move(layer), input_shape, output_shape, prng);
}

TEST_CASE("Gradcheck", "[gradcheck]")
{
  SECTION("Activation Layer")
  {
    auto layer_def = std::make_shared<kann::ActivationLayerDef>();
    auto shape = tensor::Shape::make(11, 23, 12);

    SECTION("Identity")
    {
      layer_def->type = kann::ActivationLayerDef::Type::IDENTITY;
      test_layer_def(layer_def, shape, shape);
    }

    SECTION("Sigmoid")
    {
      layer_def->type = kann::ActivationLayerDef::Type::SIGMOID;
      test_layer_def(layer_def, shape, shape);
    }

    SECTION("Tanh")
    {
      layer_def->type = kann::ActivationLayerDef::Type::TANH;
      test_layer_def(layer_def, shape, shape);
    }
  }

  SECTION("Convolutional Layer")
  {
    auto layer_def = std::make_shared<kann::ConvolutionalLayerDef>();
    layer_def->input_channel_count = 3;
    layer_def->output_channel_count = 5;
    layer_def->input_size = tensor::Vec2(8, 12);
    layer_def->output_size = tensor::Vec2(4, 6);
    layer_def->kernel_size = tensor::Vec2(5, 7);
    test_layer_def(layer_def, tensor::Shape::make(3, 8, 12), tensor::Shape::make(5, 4, 6));
  }

  SECTION("Dense Layer")
  {
    auto layer_def = std::make_shared<kann::DenseLayerDef>();
    layer_def->input_size  = 23;
    layer_def->output_size = 41;
    test_layer_def(layer_def, tensor::Shape::make(23), tensor::Shape::make(41));
  }

  SECTION("Softmax")
  {
    auto layer_def = std::make_shared<kann::SoftMaxLayerDef>();
    auto shape = tensor::Shape::make(7, 5, 6);
    test_layer_def(layer_def, shape, shape);
  }
}

int main(int argc, char* argv[])
{
  kann::initialize();

  Catch::Session session;
  session.applyCommandLine(argc, argv);
  return session.run();
}
