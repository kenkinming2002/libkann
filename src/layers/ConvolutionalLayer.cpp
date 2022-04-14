#include <libkann/layers/ConvolutionalLayer.hpp>

#include <libkann/operations/IdentityOperation.hpp>
#include <libkann/operations/ConvolutionOperation.hpp>
#include <libkann/operations/ReduceOperation.hpp>

namespace kann
{
  ConvolutionalLayer::ConvolutionalLayer(size_t input_width, size_t input_height, size_t kernel_size, size_t input_channel_count, size_t output_channel_count)
    : m_input_width(input_width), m_input_height(input_height),
      m_kernel_size(kernel_size),
      m_input_channel_count(input_channel_count), m_output_channel_count(output_channel_count) {}

  std::shared_ptr<Layer> ConvolutionalLayer::clone() const
  {
    return std::make_shared<ConvolutionalLayer>(*this);
  }

  void ConvolutionalLayer::randomize(std::default_random_engine& engine)
  {
    m_kernels.clear();
    for(size_t i=0; i<this->parameters_count(); ++i)
      m_kernels.push_back(Layer::create_tensor_gaussian(m_kernel_size * m_kernel_size, 0.0, 1.0 / m_kernel_size, engine));
  }

  size_t ConvolutionalLayer::input_size() const
  {
    return m_input_width * m_input_height * m_input_channel_count;
  }

  size_t ConvolutionalLayer::output_size() const
  {
    return (m_input_width-m_kernel_size+1) * (m_input_height-m_kernel_size+1) * m_output_channel_count;
  }

  size_t ConvolutionalLayer::parameters_count() const
  {
    return m_input_channel_count * m_output_channel_count;
  }

  std::vector<size_t> ConvolutionalLayer::parameter_sizes() const
  {
    return std::vector(this->parameters_count(), m_kernel_size * m_kernel_size);
  }

  std::vector<std::shared_ptr<const Tensor>> ConvolutionalLayer::get_parameters() const
  {
    assert(m_kernels.size() == this->parameters_count());
    return m_kernels;
  }

  void ConvolutionalLayer::set_parameters(std::vector<std::shared_ptr<const Tensor>> values)
  {
    assert(values.size() == this->parameters_count());
    m_kernels = std::move(values);
  }

  Layer::ProcessOutput ConvolutionalLayer::process(ProcessInput input) const
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
