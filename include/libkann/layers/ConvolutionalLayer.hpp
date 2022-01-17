#pragma once

#include <functional>
#include <libkann/export.hpp>
#include <libkann/layers/Layer.hpp>
#include <libkann/serialization/Eigen.hpp>

#include <Eigen/Eigen>

#include <cereal/types/vector.hpp>

#include <random>
#include <assert.h>

namespace kann
{
  class ConvolutionalLayer : public Layer
  {
  public:
    LIBKANN_SYMEXPORT ConvolutionalLayer() = default;
    LIBKANN_SYMEXPORT ConvolutionalLayer(size_t inputWidth, size_t inputHeight, size_t kernelSize, size_t inputChannelCount, size_t outputChannelCount);

  public:
    LIBKANN_SYMEXPORT std::unique_ptr<Layer> clone() const override;

  public:
    LIBKANN_SYMEXPORT size_t inputSize() const override;
    LIBKANN_SYMEXPORT size_t outputSize() const override;

  public:
    LIBKANN_SYMEXPORT std::vector<std::shared_ptr<const Variable>> parametersVariables() const override { assert(false && "Unimplmented"); }
    LIBKANN_SYMEXPORT std::vector<std::reference_wrapper<const Tensor>> parameters() const override { assert(false && "Unimplmented"); }
    LIBKANN_SYMEXPORT std::vector<std::reference_wrapper<Tensor>> parameters() override { assert(false && "Unimplmented"); }

  public:
    LIBKANN_SYMEXPORT std::pair<std::shared_ptr<const Variable>, StateVariables> operator()(std::shared_ptr<const Variable> input, StateVariables state) const override;

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(cereal::base_class<Layer>(this));

      archive(m_inputWidth, m_inputHeight);
      archive(m_kernelSize);
      archive(m_inputChannelCount, m_outputChannelCount);

      archive(m_kernels);
      archive(m_kernelsGradient);
    }

  private:
    size_t m_inputWidth,  m_inputHeight;
    size_t m_kernelSize;
    size_t m_inputChannelCount, m_outputChannelCount;

    std::vector<Eigen::MatrixXd> m_kernels;
    std::vector<Eigen::MatrixXd> m_kernelsGradient;
  };
}

CEREAL_REGISTER_TYPE(kann::ConvolutionalLayer);
CEREAL_REGISTER_POLYMORPHIC_RELATION(kann::Layer, kann::ConvolutionalLayer);
