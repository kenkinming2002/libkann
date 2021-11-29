#include <libkann/WeightLayer.hpp>

#include <iostream>

WeightLayer::WeightLayer(size_t inputSize, size_t outputSize)
{
  m_input          = Eigen::VectorXd(inputSize);
  m_weight         = Eigen::MatrixXd(outputSize, inputSize);
  m_weightGradient = Eigen::MatrixXd::Zero(outputSize, inputSize);
}

Eigen::VectorXd WeightLayer::feedForward(Eigen::VectorXd input)
{
  m_input = input;
  return m_weight * input;
}

Eigen::RowVectorXd WeightLayer::backPropagate(const Eigen::RowVectorXd& outputGradient)
{
  m_weightGradient += (m_input * outputGradient).transpose();
  return outputGradient * m_weight;
}

size_t WeightLayer::inputSize() const
{
  return m_weight.cols();
}

size_t WeightLayer::outputSize() const
{
  return m_weight.rows();
}

void WeightLayer::train(double learningRate)
{
  auto norm = (learningRate * m_weightGradient).norm();
  if(norm < 0.00000001f)
    std::cout << "Warning: vanishing gradient detected" << std::endl;

  if(norm > 100000.0f)
    std::cout << "Warning: exploding gradient detected" << std::endl;

  m_weight -= learningRate * m_weightGradient;
  m_weightGradient.setZero();
}

template<typename PRNG>
void WeightLayer::randomize(PRNG& prng)
{
  std::uniform_real_distribution<double> weightDistribution(-1.0, 1.0);
  m_weight = Eigen::MatrixXd::NullaryExpr(m_weightGradient.rows(), m_weight.cols(), [&](){
    return weightDistribution(prng);
  });
}
template void WeightLayer::randomize(std::mt19937&);

template<typename PRNG>
WeightLayer WeightLayer::cross(const WeightLayer& lhs, const WeightLayer& rhs, PRNG& prng, double mutationRate)
{
  assert(lhs.m_weight.size() != 0);
  assert(lhs.m_weight.size() == rhs.m_weight.size());

  std::uniform_int_distribution<> distribution(0, 1);
  std::uniform_real_distribution<double> mutationDistribution(0.0, 1.0);
  std::uniform_real_distribution<double> weightDistribution(-1.0, 1.0);

  WeightLayer result = lhs;;
  for(long i=0; i<result.m_weight.size(); ++i)
  {
    result.m_weight.data()[i] = distribution(prng) == 0 ? lhs.m_weight.data()[i] : rhs.m_weight.data()[i];
    if(mutationDistribution(prng) < mutationRate)
      result.m_weight.data()[i] = weightDistribution(prng);
  }

  return result;
}
template WeightLayer WeightLayer::cross(const WeightLayer&, const WeightLayer&, std::mt19937&, double);
