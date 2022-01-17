#include <libkann/layers/Layer.hpp>
#include <random>

namespace kann
{
  void randomize(Layer& layer, std::default_random_engine& engine)
  {
    const double range = std::sqrt(2.0 / layer.inputSize());
    std::uniform_real_distribution distWeight(-range, range);

    auto parameters = layer.parameters();
    for(std::reference_wrapper<Tensor> parameter : parameters)
      parameter.get().asArray() = Eigen::ArrayXd::NullaryExpr(parameter.get().size(), [&](){
        return distWeight(engine);
      });
  }

  std::unique_ptr<Layer> cross(const Layer& lhs, const Layer& rhs, std::default_random_engine& engine, double mutationRate)
  {
    std::unique_ptr<Layer> result = lhs.clone();

    const auto lhsParameters = lhs.parameters();
    const auto rhsParameters = rhs.parameters();
    auto parameters = result->parameters();

    const double range = std::sqrt(2.0 / result->inputSize());
    std::uniform_real_distribution distWeight(-range, range);
    std::uniform_real_distribution distMutation(0.0,1.0);
    std::uniform_int_distribution distSelection(0,1);

    assert(parameters.size() == lhsParameters.size());
    assert(parameters.size() == rhsParameters.size());
    for(size_t i=0; i<parameters.size(); ++i)
    {
      parameters[i].get().asArray() = lhsParameters[i].get().asArray().binaryExpr(rhsParameters[i].get().asArray(), [&](double a, double b){
        if(distMutation(engine)>=mutationRate)
          return distWeight(engine);
        else
          return distSelection(engine) == 0 ? a : b;
      });
    }

    return result;
  }
}
