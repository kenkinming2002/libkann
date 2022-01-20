#include <libkann/layers/Layer.hpp>
#include <random>

namespace kann
{
  void randomize(Layer& layer, std::default_random_engine& engine)
  {
    const double range = std::sqrt(2.0 / layer.inputSize());
    std::uniform_real_distribution distWeight(-range, range);

    auto parameters = layer.parameters(TAG_ALL);
    for(std::reference_wrapper parameter : parameters)
    {
      auto newParameter = std::make_shared<Tensor>(parameter.get()->size());
      newParameter->asArray() = Eigen::ArrayXd::NullaryExpr(newParameter->size(), [&](){
        return distWeight(engine);
      });
      parameter.get() = newParameter;
    }
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

  static void cross(
      std::vector<std::reference_wrapper<const std::shared_ptr<const Tensor>>> lhs,
      std::vector<std::reference_wrapper<const std::shared_ptr<const Tensor>>> rhs,
      std::vector<std::reference_wrapper<std::shared_ptr<const Tensor>>> result,
      std::default_random_engine& engine, double mutationRate, double range)
  {
    assert(lhs.size() == rhs.size());
    for(size_t i=0; i<result.size(); ++i)
      result[i].get() = cross(*lhs[i].get(), *rhs[i].get(), engine, mutationRate, range);
  }

  std::unique_ptr<Layer> cross(const Layer& lhs, const Layer& rhs, std::default_random_engine& engine, double mutationRate)
  {
    auto result = lhs.clone();

    assert(lhs.inputSize() == rhs.inputSize());
    const double range = std::sqrt(2.0 / lhs.inputSize());

    const auto lhsParameters = lhs.parameters(TAG_ALL);
    const auto rhsParameters = rhs.parameters(TAG_ALL);
    auto parameters = result->parameters(TAG_ALL);
    cross(lhsParameters, rhsParameters, parameters, engine, mutationRate, range);

    return result;
  }
}
