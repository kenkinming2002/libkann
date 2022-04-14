#include <libkann/layers/WeightLayer.hpp>

#include <libkann/operations/MatrixMultiplyOperation.hpp>
#include <libkann/operations/ReduceOperation.hpp>

namespace kann
{
  WeightLayer::WeightLayer(size_t input_size, size_t output_size)
    : m_input_size(input_size), m_output_size(output_size) {}

  std::shared_ptr<Layer> WeightLayer::clone() const
  {
    return std::make_shared<WeightLayer>(*this);
  }

  void WeightLayer::randomize(std::default_random_engine& engine)
  {
    m_weight = Layer::create_tensor_gaussian(m_input_size * m_output_size, 0.0, 1.0 / std::sqrt(m_input_size), engine);
    m_bias   = Layer::create_tensor_gaussian(m_output_size               , 0.0, 0.0 / std::sqrt(m_input_size), engine);
  }

  size_t WeightLayer::input_size() const
  {
    return m_input_size;
  }

  size_t WeightLayer::output_size() const
  {
    return m_output_size;
  }

  size_t WeightLayer::parameters_count() const
  {
    return 2;
  }

  std::vector<size_t> WeightLayer::parameter_sizes() const
  {
    return {m_input_size * m_output_size, m_output_size};
  }

  std::vector<std::shared_ptr<const Tensor>> WeightLayer::get_parameters() const
  {
    assert(m_weight);
    assert(m_bias);
    return {m_weight, m_bias};
  }

  void WeightLayer::set_parameters(std::vector<std::shared_ptr<const Tensor>> values)
  {
    assert(values.size() == 2);
    m_weight = std::move(values[0]);
    m_bias   = std::move(values[1]);
  }

  Layer::ProcessOutput WeightLayer::process(ProcessInput input) const
  {
    ProcessOutput output;

    auto weight = input.parameters[0];
    auto bias   = input.parameters[1];

    // TODO: Fuse them into a single operation
    auto prod = Variable::apply(
        MatrixMultiplyOperation(m_output_size, 1, m_input_size, false, false),
        {std::move(weight), std::move(input.variable)}
    );
    output.variable = Variable::apply(ReduceOperation(2), {std::move(bias), std::move(prod)});
    output.states = std::move(input.states);
    return output;
  }

}
