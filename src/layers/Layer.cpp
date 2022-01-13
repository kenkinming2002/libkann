#include <libkann/layers/Layer.hpp>

namespace kann
{
  void Layer::input(Eigen::VectorXd input)
  {
    m_input = input;
  }

  void Layer::outputGradient(Eigen::VectorXd outputGradient)
  {
    m_outputGradient = outputGradient;
  }

  const Eigen::VectorXd& Layer::input() const
  {
    return m_input;
  }

  const Eigen::VectorXd& Layer::outputGradient() const
  {
    return m_outputGradient;
  }

  std::unique_ptr<Layer> Layer::cross(const Layer& lhs, const Layer& rhs, std::default_random_engine& engine, double mutationRate)
  {
    auto result = lhs.clone();

    assert(lhs.inputSize() == result->inputSize());
    assert(rhs.inputSize() == result->inputSize());

    const auto range = std::sqrt(2.0 / result->inputSize());
    std::uniform_int_distribution<int> distSelection(0, 1);
    std::uniform_real_distribution<double> distMutation(0.0, 1.0);
    std::uniform_real_distribution<double> distWeight(-range, range);

    const auto lhsParams = lhs.params();
    const auto rhsParams = rhs.params();
    auto params = result->params();

    assert(lhsParams.size() == params.size());
    assert(rhsParams.size() == params.size());

    for(size_t i=0; i<params.size(); ++i)
    {
      auto lhsParam = Eigen::Map<const Eigen::ArrayXd>(lhsParams[i].data(), lhsParams[i].size());
      auto rhsParam = Eigen::Map<const Eigen::ArrayXd>(rhsParams[i].data(), rhsParams[i].size());
      auto param = Eigen::Map<Eigen::ArrayXd>(params[i].data(), params[i].size());

      std::cout << "LHS size:" << lhsParam.size() << ", RHS size:" << rhsParam.size() << std::endl;
    }

    for(size_t i=0; i<params.size(); ++i)
    {
      auto lhsParam = Eigen::Map<const Eigen::ArrayXd>(lhsParams[i].data(), lhsParams[i].size());
      auto rhsParam = Eigen::Map<const Eigen::ArrayXd>(rhsParams[i].data(), rhsParams[i].size());
      auto param = Eigen::Map<Eigen::ArrayXd>(params[i].data(), params[i].size());

      std::cout << "LHS size:" << lhsParam.size() << std::endl;
      std::cout << "RHS size:" << rhsParam.size() << std::endl;

      assert(lhsParam.size() == rhsParam.size());
      param = lhsParam.binaryExpr(rhsParam, [&](auto a, auto b){
        if(distMutation(engine) > mutationRate)
          return distWeight(engine);

        return distSelection(engine) == 0 ? a : b;
      });
    }

    return result;
  }

  void Layer::randomize(std::default_random_engine& engine)
  {
    const double range = std::sqrt(2.0 / this->inputSize());
    std::uniform_real_distribution<double> distWeight(-range, range);
    auto params = this->params();
    for(size_t i=0; i<params.size(); ++i)
    {
      auto param = Eigen::Map<Eigen::ArrayXd>(params[i].data(), params[i].size());
      param = Eigen::ArrayXd::NullaryExpr(param.size(), [&](){return distWeight(engine);});
    }
  }

  void Layer::train(double learningRate, unsigned /*tags*/)
  {
    auto params = this->params();
    auto paramsGradient = this->paramsGradient();
    assert(params.size() == paramsGradient.size());
    for(size_t i=0; i<params.size(); ++i)
    {
      auto param         = Eigen::Map<Eigen::ArrayXd>(params[i].data(), params[i].size());
      auto paramGradient = Eigen::Map<Eigen::ArrayXd>(paramsGradient[i].data(), paramsGradient[i].size());

      param -= learningRate * paramGradient;
      paramGradient.setZero();
    }
  }
}
