#include <libkann/layers/Layer.hpp>

namespace kann
{
  Layer::Layer(size_t paramsCount)
    : m_params(Eigen::ArrayXd::Zero(paramsCount)) {}

  void Layer::randomize(std::default_random_engine& engine)
  {
    const double range = std::sqrt(2.0 / this->inputSize());
    std::uniform_real_distribution<double> distWeight(-range, range);
    for(size_t i=0; i<m_params.size(); ++i)
      m_params(i) = distWeight(engine);
  }

  std::unique_ptr<Layer> Layer::cross(const Layer& lhs, const Layer& rhs, std::default_random_engine& engine, double mutationRate)
  {
    assert(lhs.inputSize() == rhs.inputSize());
    assert(lhs.outputSize() == rhs.outputSize());
    assert(lhs.m_params.size() == rhs.m_params.size());

    assert(lhs.inputSize() != 0);
    assert(lhs.outputSize() != 0);

    auto result = lhs.clone();

    const auto range = std::sqrt(2.0 / lhs.inputSize());
    std::uniform_int_distribution<int> distSelection(0, 1);
    std::uniform_real_distribution<double> distMutation(0.0, 1.0);
    std::uniform_real_distribution<double> distWeight(-range, range);

    for(size_t i=0; i<result->m_params.size(); ++i)
      if(distMutation(engine) > mutationRate)
        result->m_params(i) = distWeight(engine);
      else if(distSelection(engine) == 1)
        result->m_params(i) =  rhs.m_params(i);

    return result;
  }

  void Layer::train(double learningRate, const Eigen::ArrayXd& layerGradient)
  {
    m_params -= learningRate * layerGradient;
  }
}
