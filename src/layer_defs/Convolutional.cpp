#include <libkann/layer_defs/Convolutional.hpp>

#include <libkann/Layer.hpp>

#include <libkann/operations/IdentityOperation.hpp>
#include <libkann/operations/ConvolutionOperation.hpp>
#include <libkann/operations/ReduceOperation.hpp>

#include <range/v3/all.hpp>

namespace kann
{
  ConvolutionalLayerDef::ConvolutionalLayerDef(size_t input_width, size_t input_height, size_t kernel_size, size_t input_channel_count, size_t output_channel_count)
    : m_input_width(input_width), m_input_height(input_height),
      m_kernel_size(kernel_size),
      m_input_channel_count(input_channel_count), m_output_channel_count(output_channel_count) {}

  std::shared_ptr<Layer> ConvolutionalLayerDef::create(std::default_random_engine& prng) const
  {
    auto layer = std::make_shared<Layer>();
    layer->def = shared_from_this();
    layer->parameters = ranges::views::generate_n([&](){
      return std::make_shared<const Tensor>(Tensor::gaussian(m_kernel_size * m_kernel_size, prng, 0.0, 1.0 / m_kernel_size));
    }, this->parameters_count()) | ranges::to_vector;
    return layer;
  }

  size_t ConvolutionalLayerDef::input_size() const
  {
    return m_input_width * m_input_height * m_input_channel_count;
  }

  size_t ConvolutionalLayerDef::output_size() const
  {
    return (m_input_width-m_kernel_size+1) * (m_input_height-m_kernel_size+1) * m_output_channel_count;
  }

  size_t ConvolutionalLayerDef::parameters_count() const
  {
    return m_input_channel_count * m_output_channel_count;
  }

  std::vector<size_t> ConvolutionalLayerDef::parameters_sizes() const
  {
    return ranges::views::repeat_n(m_kernel_size * m_kernel_size, this->parameters_count()) | ranges::to_vector;
  }

  LayerDef::ProcessOutput ConvolutionalLayerDef::process(ProcessInput input) const
  {
    // 1: Split input into channels
    std::vector<std::shared_ptr<const Variable>> input_channels;
    {
      input_channels.reserve(m_input_channel_count);
      for(size_t i=0; i<m_input_channel_count; ++i)
        input_channels.push_back(Variable::apply(
            IdentityOperation(
              m_input_width * m_input_height * m_input_channel_count,
              m_input_width * m_input_height,
              m_input_width * m_input_height * i
            ),
            {input.variable}
        ));
    }

    // 2: Perform Convolution
    std::vector<std::shared_ptr<const Variable>> results;
    results.reserve(this->parameters_count());
    for(size_t j=0; j<m_output_channel_count; ++j)
      for(size_t i=0; i<m_input_channel_count; ++i)
      {
        auto input_channel = input_channels[i];
        auto kernel        = input.parameters[j * m_input_channel_count + i];
        results.push_back(Variable::apply(
            ConvolutionOperation(m_input_width, m_input_height, m_kernel_size),
            {input_channel, kernel}
        ));
      }

    // 3: Reduce result to obtain output channel variables
    std::vector<std::shared_ptr<const Variable>> output_channels;
    for(size_t j=0; j<m_output_channel_count; ++j)
      output_channels.push_back(Variable::apply(
          ReduceOperation(m_input_channel_count),
          std::vector(
            &results[m_input_channel_count * j],
            &results[m_input_channel_count * (j+1)]
          )
      ));

    // 4: Concat output channel variables
    const size_t output_width  = m_input_width  - m_kernel_size + 1;
    const size_t output_height = m_input_height - m_kernel_size + 1;

    for(size_t i=0; i<m_output_channel_count; ++i)
      output_channels[i] = Variable::apply(
        IdentityOperation(
          output_width * output_height,
          output_width * output_height * m_output_channel_count,
          output_width * output_height * i
        ),
        {std::move(output_channels[i])}
      );

    ProcessOutput output;
    output.variable = Variable::apply(ReduceOperation(m_output_channel_count), std::move(output_channels));
    output.states   = std::move(input.states);
    return output;
  }
}
