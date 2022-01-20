#include <libkann/layers/DeconvolutionalLayer.hpp>

#include <libkann/operations/IdentityOperation.hpp>
#include <libkann/operations/DeconvolutionOperation.hpp>
#include <libkann/operations/ReduceOperation.hpp>

#include <assert.h>

namespace kann
{
  DeconvolutionalLayer::DeconvolutionalLayer(size_t inputWidth, size_t inputHeight, size_t kernelSize, size_t inputChannelCount, size_t outputChannelCount)
    : m_inputWidth(inputWidth), m_inputHeight(inputHeight),
      m_kernelSize(kernelSize),
      m_inputChannelCount(inputChannelCount), m_outputChannelCount(outputChannelCount)
  {

    initializeVariables();

    m_kernels.resize(m_inputChannelCount * m_outputChannelCount);
    std::generate(m_kernels.begin(), m_kernels.end(), [this](){
      return std::make_shared<const Tensor>(m_kernelSize * m_kernelSize);
    });
  }

  void DeconvolutionalLayer::initializeVariables()
  {
    m_kernelsVariable.resize(m_inputChannelCount * m_outputChannelCount);
    std::generate(m_kernelsVariable.begin(), m_kernelsVariable.end(), [](){
      return std::make_shared<const Variable>();
    });
  }

  std::unique_ptr<Layer> DeconvolutionalLayer::clone() const
  {
    return std::make_unique<DeconvolutionalLayer>(*this);
  }

  size_t DeconvolutionalLayer::inputSize() const
  {
    return m_inputWidth * m_inputHeight * m_inputChannelCount;
  }

  size_t DeconvolutionalLayer::outputSize() const
  {
    return (m_inputWidth+m_kernelSize-1) * (m_inputHeight+m_kernelSize-1) * m_outputChannelCount;
  }

  auto DeconvolutionalLayer::operator()(std::shared_ptr<const Variable> input, StateVariables state) const -> std::pair<std::shared_ptr<const Variable>, StateVariables>
  {
    /* TODO: Consider marking them as override final */
    const size_t outputWidth = m_inputWidth + m_kernelSize - 1;
    const size_t outputHeight = m_inputHeight + m_kernelSize - 1;

    // Split input into channels
    std::vector<std::shared_ptr<const Variable>> inputChannels(m_inputChannelCount);
    for(size_t i=0; i<m_inputChannelCount; ++i)
      inputChannels[i] = std::make_shared<const Variable>(
        std::vector{input},
        std::make_shared<IdentityOperation>(
          m_inputWidth * m_inputHeight * m_inputChannelCount,
          m_inputWidth * m_inputHeight,
          m_inputWidth * m_inputHeight * i
        )
      );

    std::vector<std::shared_ptr<const Variable>> outputChannels(m_outputChannelCount);
    for(size_t i=0; i<m_outputChannelCount; ++i)
    {
      std::vector<std::shared_ptr<const Variable>> results(m_inputChannelCount);
      for(size_t j=0; j<m_inputChannelCount; ++j)
        results[j] = std::make_shared<const Variable>(
          std::vector{inputChannels[j], m_kernelsVariable[i * m_inputChannelCount + j]},
          std::make_shared<DeconvolutionOperation>(m_inputWidth, m_inputHeight, m_kernelSize)
        );

      outputChannels[i] = std::make_shared<const Variable>(
        std::move(results),
        std::make_shared<ReduceOperation>(m_inputChannelCount)
      );
    }

    // Concat output channels
    for(size_t i=0; i<m_outputChannelCount; ++i)
      outputChannels[i] = std::make_shared<const Variable>(
        std::vector{std::move(outputChannels[i])},
        std::make_shared<IdentityOperation>(
          outputWidth * outputHeight,
          outputWidth * outputHeight * m_outputChannelCount,
          outputWidth * outputHeight * i
        )
      );

    auto output = std::make_shared<const Variable>(
        std::move(outputChannels),
        std::make_shared<ReduceOperation>(m_outputChannelCount)
    );

    return std::make_pair(std::move(output), std::move(state));
  }

  std::vector<std::shared_ptr<const Variable>> DeconvolutionalLayer::parametersVariables(unsigned tags) const
  {
    return m_kernelsVariable;
  }

  std::vector<std::reference_wrapper<const std::shared_ptr<const Tensor>>> DeconvolutionalLayer::parameters(unsigned tags) const
  {
    std::vector<std::reference_wrapper<const std::shared_ptr<const Tensor>>> result;
    result.reserve(m_kernels.size());
    for(const auto& kernel : m_kernels)
      result.push_back(kernel);

    return result;
  }

  std::vector<std::reference_wrapper<std::shared_ptr<const Tensor>>> DeconvolutionalLayer::parameters(unsigned tags)
  {
    std::vector<std::reference_wrapper<std::shared_ptr<const Tensor>>> result;
    result.reserve(m_kernels.size());
    for(auto& kernel : m_kernels)
      result.push_back(kernel);

    return result;
  }
}
