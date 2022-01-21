#include <libkann/layers/Layer.hpp>

#include <random>

namespace kann
{
  void randomize(Layer& layer, std::default_random_engine& engine)
  {
    const double range = std::sqrt(2.0 / layer.inputSize());
    std::uniform_real_distribution distWeight(-range, range);

    for(const auto& parameter : layer.parameters(TAG_ALL))
    {
      const size_t size = parameter->value->size();
      auto newValue = std::make_shared<Tensor>(size);
      newValue->asArray() = Eigen::ArrayXd::NullaryExpr(size, [&](){
        return distWeight(engine);
      });
      parameter->value = std::move(newValue);
    }
  }

  std::unique_ptr<Layer> cross(const Layer& lhs, const Layer& rhs, std::default_random_engine& engine, double mutationRate)
  {
    auto result = lhs.clone();

    assert(lhs.inputSize() == rhs.inputSize());
    const double range = std::sqrt(2.0 / lhs.inputSize());

    std::uniform_real_distribution distWeight(-range, range);
    std::uniform_real_distribution distMutation(0.0,1.0);
    std::uniform_int_distribution distSelection(0,1);

    const auto lhsParameters = lhs.parameters(TAG_ALL);
    const auto rhsParameters = rhs.parameters(TAG_ALL);
    auto resultParameters = result->parameters(TAG_ALL);

    assert(resultParameters.size() == lhsParameters.size());
    assert(resultParameters.size() == rhsParameters.size());
    for(size_t i=0; i<resultParameters.size(); ++i)
    {
      const auto& lhsValue = lhsParameters[i]->value;
      const auto& rhsValue = rhsParameters[i]->value;
      auto& resultValue = resultParameters[i]->value;

      assert(resultValue->size() == lhsValue->size());
      assert(resultValue->size() == rhsValue->size());

      const size_t size = resultValue->size();
      auto newValue = std::make_shared<Tensor>(size);
      newValue->asArray() = lhsValue->asArray().binaryExpr(rhsValue->asArray(), [&](double a, double b){
        if(distMutation(engine)>=mutationRate)
          return distWeight(engine);
        else
          return distSelection(engine) == 0 ? a : b;
      });
      resultValue = std::move(newValue);
    }

    return result;
  }
}
