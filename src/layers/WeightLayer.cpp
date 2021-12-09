#include <libkann/layers/WeightLayer.hpp>

#include <iostream>

namespace kann
{
  WeightLayer::WeightLayer(size_t inputSize, size_t outputSize)
  {
    m_weight         = Eigen::MatrixXd(outputSize, inputSize);
    m_weightGradient = Eigen::MatrixXd::Zero(outputSize, inputSize);
  }

  size_t WeightLayer::inputSize() const
  {
    return m_weight.cols();
  }

  size_t WeightLayer::outputSize() const
  {
    return m_weight.rows();
  }

  void WeightLayer::randomize(std::default_random_engine& engine)
  {
    std::uniform_real_distribution<double> weightDistribution(-1.0, 1.0);
    m_weight = Eigen::MatrixXd::NullaryExpr(m_weightGradient.rows(), m_weight.cols(), [&](){
      return weightDistribution(engine);
    });
  }

  Eigen::VectorXd WeightLayer::feedForward()
  {
    return m_weight * this->input();
  }

  Eigen::RowVectorXd WeightLayer::backPropagate(const Eigen::RowVectorXd& outputGradient)
  {
    m_weightGradient += (this->input() * outputGradient).transpose();
    return outputGradient * m_weight;
  }

  void WeightLayer::train(double learningRate)
  {
    m_weight -= learningRate * m_weightGradient;
    m_weightGradient.setZero();
  }

  std::unique_ptr<Layer> WeightLayer::cross(const Layer& _other, std::default_random_engine& engine, double mutationRate) const
  {
    auto result = std::make_unique<WeightLayer>(*this);
    const auto& other = dynamic_cast<const WeightLayer&>(_other);

    std::uniform_int_distribution<> distribution(0, 1);
    std::uniform_real_distribution<double> mutationDistribution(0.0, 1.0);
    std::uniform_real_distribution<double> weightDistribution(-1.0, 1.0);

    assert(m_weight.size() != 0);
    assert(m_weight.size() == other.m_weight.size());
    for(long i=0; i<result->m_weight.size(); ++i)
    {
      result->m_weight.data()[i] = distribution(engine) == 0 ? m_weight.data()[i] : other.m_weight.data()[i];
      if(mutationDistribution(engine) < mutationRate)
        result->m_weight.data()[i] = weightDistribution(engine);
    }

    return result;
  }
}
