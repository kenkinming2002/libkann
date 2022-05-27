#include <libkann/layer_defs/Weight.hpp>

#include <libkann/Layer.hpp>
#include <libkann/Variable.hpp>

#include <libkann/operations/MatrixMultiplyOperation.hpp>
#include <libkann/operations/ReduceOperation.hpp>

namespace kann
{
  WeightLayerDef::WeightLayerDef(size_t input_size, size_t output_size)
    : m_input_size(input_size), m_output_size(output_size) {}

  std::shared_ptr<Layer> WeightLayerDef::create(std::default_random_engine& prng) const
  {
    auto layer = std::make_shared<Layer>();
    layer->def = shared_from_this();
    layer->parameters = {
      std::make_shared<const Tensor>(Tensor::gaussian(m_input_size * m_output_size, prng, 0.0, 1.0 / std::sqrt(m_input_size))),
      std::make_shared<const Tensor>(Tensor::gaussian(m_output_size               , prng, 0.0, 1.0 / std::sqrt(m_input_size)))
    };
    return layer;
  }

  size_t WeightLayerDef::input_size() const
  {
    return m_input_size;
  }

  size_t WeightLayerDef::output_size() const
  {
    return m_output_size;
  }

  LayerDef::ProcessOutput WeightLayerDef::process(ProcessInput input) const
  {
    ProcessOutput output;

    auto weight = input.parameters[0];
    auto bias   = input.parameters[1];

    // TODO: Fuse them into a single operation
    auto product = Variable::apply(MatrixMultiplyOperation(m_output_size, 1, m_input_size, false, false), {weight, input.variable});
    output.variable = Variable::apply(ReduceOperation(2), {product, bias});

    return output;
  }

  size_t WeightLayerDef::parameters_count() const
  {
    return 2;
  }

  std::vector<size_t> WeightLayerDef::parameters_sizes() const
  {
    return {m_input_size * m_output_size, m_output_size};
  }
}
