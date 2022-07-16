#include <catch2/catch_all.hpp>
#include <catch2/catch_session.hpp>

#include <libkann/Initialize.hpp>
#include <libtensor/Map.hpp>

#include <libkann/Layer.hpp>
#include <libkann/LayerDef.hpp>
#include <libkann/LayerStorage.hpp>

#include <libkann/layer_defs/Activation.hpp>
#include <libkann/layer_defs/Convolutional.hpp>
#include <libkann/layer_defs/Dense.hpp>
#include <libkann/layer_defs/SoftMax.hpp>

tensor::Tensor<const float> with_unit_batching(tensor::Tensor<const float> value)
{
  tensor::Shape shape = value.shape();
  return std::move(value).reshape(tensor::Shape::make(1, shape));
}

tensor::Tensor<const float> without_unit_batching(tensor::Tensor<const float> value)
{
  tensor::Shape shape = value.shape();

  // No need to check explicitly as the assertion in reshape would catch any size mismatch
  // if the first dimension either does not exist or is not 1
  return std::move(value).reshape(shape.drop_front(1));
}

struct LayerDerivative
{
  tensor::Tensor<float> input;
  std::vector<tensor::Tensor<float>> parameters;

  static LayerDerivative create_uninitialized_for(const kann::Layer& layer)
  {
    const tensor::Shape input_shape  = layer.def->get_input_shape();
    const tensor::Shape output_shape = layer.def->get_output_shape();

    LayerDerivative derivative{ .input = tensor::Tensor<float>::create(tensor::Shape::make(output_shape, input_shape))};
    for(const kann::Variable* parameter : layer.storage->get_parameters())
    {
      const tensor::Shape parameter_shape = parameter->value.shape();
      derivative.parameters.push_back(tensor::Tensor<float>::create(tensor::Shape::make(output_shape, parameter_shape)));
    }
    return derivative;
  }
};

tensor::Tensor<float> create_basis(tensor::Shape shape, size_t i)
{
  tensor::Tensor<float> result = tensor::Tensor<float>::create(shape);
  std::fill_n(result.data(), result.size(), 0.0f);
  result.data()[i] = 1.0f;
  return result;
}

inline LayerDerivative compute_analytical_derivative(kann::Layer& layer, const tensor::Tensor<const float>& random_input)
{
  const tensor::Shape input_shape  = layer.def->get_input_shape();
  const tensor::Shape output_shape = layer.def->get_output_shape();

  const size_t input_size  = input_shape.size();
  const size_t output_size = output_shape.size();

  LayerDerivative derivative = LayerDerivative::create_uninitialized_for(layer);
  for(size_t j=0; j<output_size; ++j)
  {
    tensor::Tensor<const float> input  = with_unit_batching(random_input.clone());
    tensor::Tensor<const float> output = without_unit_batching(layer.forward(input.clone()));
    (void)output;

    tensor::Tensor<const float> output_gradient = with_unit_batching(create_basis(output_shape, j));
    tensor::Tensor<const float> input_gradient  = without_unit_batching(layer.backward(output_gradient.clone()));

    // Derivative for input
    {
      tensor::Tensor<float>       _input_derivative = derivative.input.reshape(tensor::Shape::make(output_size, input_size));
      tensor::Tensor<const float> _input_gradient   = input_gradient.reshape(tensor::Shape::make(input_size));
      for(size_t i=0; i<input_size; ++i)
        _input_derivative(j,i) = _input_gradient(i);
    }

    // Derivative for parameters
    {
      const std::vector<kann::Variable*> parameters = layer.storage->get_parameters();
      const size_t parameters_count = parameters.size();
      for(size_t k=0; k<parameters_count; ++k)
      {
        const tensor::Shape parameter_shape = parameters[k]->shape;
        const size_t parameter_size = parameter_shape.size();

        auto _parameter_derivative = derivative.parameters[k].reshape(tensor::Shape::make(output_size, parameter_size));
        auto _parameter_gradient   = parameters[k]->gradient .reshape(tensor::Shape::make(parameter_size));
        for(size_t i=0; i<parameter_size; ++i)
          _parameter_derivative(j,i) = _parameter_gradient(i);
      }
    }
  }

  return derivative;
}

tensor::Tensor<float> perturb(tensor::Tensor<float> value, size_t i, float diff)
{
  value.flatten()(i) += diff;
  return value;
}

inline LayerDerivative compute_numerical_derivative(kann::Layer& layer, const tensor::Tensor<const float>& random_input, float dx)
{
  const tensor::Shape input_shape  = layer.def->get_input_shape();
  const tensor::Shape output_shape = layer.def->get_output_shape();

  const size_t input_size  = input_shape.size();
  const size_t output_size = output_shape.size();

  LayerDerivative derivative = LayerDerivative::create_uninitialized_for(layer);

  // Derivative with respect to inputs
  {
    for(size_t i=0; i<input_size; ++i)
    {
      tensor::Tensor<const float> input1  = with_unit_batching(random_input.clone());
      tensor::Tensor<const float> output1 = without_unit_batching(layer.forward(input1.clone()));

      tensor::Tensor<const float> input2  = with_unit_batching(perturb(random_input.clone(), i, dx));
      tensor::Tensor<const float> output2 = without_unit_batching(layer.forward(input2.clone()));

      auto output_gradient = tensor::binary_map(output2, output1, [&dx](float output2, float output1){ return (output2 - output1) / dx; });

      auto _input_derivative  = derivative.input.reshape(tensor::Shape::make(output_size, input_size));
      auto _output_gradient   = output_gradient .reshape(tensor::Shape::make(output_size));
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
      const tensor::Shape parameter_shape = parameters[k]->shape;
      const size_t parameter_size = parameter_shape.size();

      // This time we perturb the parameter instead of input
      for(size_t i=0; i<parameter_size; ++i)
      {
        tensor::Tensor<const float> input1  = with_unit_batching(random_input.clone());
        tensor::Tensor<const float> output1 = without_unit_batching(layer.forward(input1.clone()));

        // Save and perturb
        auto saved_parameter = std::move(parameters[k]->value);
        parameters[k]->value = perturb(saved_parameter.clone(), i, dx);

        tensor::Tensor<const float> input2  = with_unit_batching(random_input.clone());
        tensor::Tensor<const float> output2 = without_unit_batching(layer.forward(input2.clone()));

        // Restore
        parameters[k]->value = std::move(saved_parameter);

        auto output_gradient = tensor::binary_map(output2, output1, [&dx](float output2, float output1){ return (output2 - output1) / dx; });

        auto _parameter_derivative = derivative.parameters[k].reshape(tensor::Shape::make(output_size, parameter_size));
        auto _output_gradient      = output_gradient         .reshape(tensor::Shape::make(output_size));
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
  static constexpr float DX = 0.0005f;

  auto random_input = tensor::create_uniform(layer->def->get_input_shape(), -1.0f, 1.0f, prng);

  LayerDerivative analytical = compute_analytical_derivative(*layer, random_input);
  LayerDerivative numerical  = compute_numerical_derivative(*layer, random_input, DX);

  tensor::Tensor<const float> _analytical_derivative = analytical.input.flatten();
  tensor::Tensor<const float> _numerical_derivative  = numerical.input.flatten();
  for(size_t i=0; i<_analytical_derivative.size(); ++i)
  {
    const float analytical_value = _analytical_derivative(i);
    const float numerical_value  = _numerical_derivative(i);
    REQUIRE(std::abs(analytical_value - numerical_value) <= 0.001f);
  }

  for(const auto& [analytical_derivative, numerical_derivative] : ranges::views::zip(analytical.parameters, numerical.parameters))
  {
    tensor::Tensor<const float> _analytical_derivative = analytical_derivative.flatten();
    tensor::Tensor<const float> _numerical_derivative  = numerical_derivative.flatten();
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
