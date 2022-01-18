#include <libkann/layers/Layer.hpp>
#include <random>

namespace kann
{
  void randomize(Layer& layer, std::default_random_engine& engine)
  {
    const double range = std::sqrt(2.0 / layer.inputSize());
    std::uniform_real_distribution distWeight(-range, range);

    auto parameters = layer.parameters();
    for(auto& parameter : parameters)
    {
      auto newParameter = std::make_shared<Tensor>(parameter->size());
      newParameter->asArray() = Eigen::ArrayXd::NullaryExpr(newParameter->size(), [&](){
        return distWeight(engine);
      });

      parameter = newParameter;
    }
    layer.parameters(std::move(parameters));
  }

  static std::shared_ptr<const Tensor> cross(const Tensor& lhs, const Tensor& rhs, std::default_random_engine& engine, double mutationRate, double range)
  {
    std::uniform_real_distribution distWeight(-range, range);
    std::uniform_real_distribution distMutation(0.0,1.0);
    std::uniform_int_distribution distSelection(0,1);

    auto result = std::make_shared<Tensor>(lhs.size());
    result->asArray() = lhs.asArray().binaryExpr(rhs.asArray(), [&](double a, double b){
      if(distMutation(engine)>=mutationRate)
        return distWeight(engine);
      else
        return distSelection(engine) == 0 ? a : b;
    });

    return result;
  }

  static auto cross(std::vector<std::shared_ptr<const Tensor>> lhs, std::vector<std::shared_ptr<const Tensor>> rhs, std::default_random_engine& engine, double mutationRate, double range)
  {
    assert(lhs.size() == rhs.size());
    std::vector<std::shared_ptr<const Tensor>> result(lhs.size());
    for(size_t i=0; i<result.size(); ++i)
      result[i] = cross(*lhs[i], *rhs[i], engine, mutationRate, range);

    return result;
  }

  std::unique_ptr<Layer> cross(const Layer& lhs, const Layer& rhs, std::default_random_engine& engine, double mutationRate)
  {
    assert(lhs.inputSize() == rhs.inputSize());
    const double range = std::sqrt(2.0 / lhs.inputSize());

    const auto lhsParameters = lhs.parameters();
    const auto rhsParameters = rhs.parameters();
    auto parameters = cross(lhsParameters, rhsParameters, engine, mutationRate, range);

    std::unique_ptr<Layer> result = lhs.clone();
    result->parameters(std::move(parameters));
    return result;
  }
}
