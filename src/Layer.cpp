#include <libkann/Layer.hpp>

namespace kann
{
  std::shared_ptr<const Tensor> Layer::create_tensor_gaussian(size_t size, double mean, double variance, std::default_random_engine& engine)
  {
    std::normal_distribution dist(mean, variance);
    return std::make_shared<const Tensor>(Tensor::nullaryExpr(size, [&](){ return dist(engine); }));
  }

  std::shared_ptr<Layer> cross(const Layer& lhs, const Layer& rhs, std::default_random_engine& engine, double mutation_rate)
  {
    auto layer = lhs.clone();
    layer->randomize(engine);

    // Randomize and mutate parameters
    auto parameters = layer->get_parameters();
    {
      std::uniform_real_distribution dist_mutation(0.0,1.0);
      std::uniform_int_distribution  dist_selection(0,1);

      const auto lhs_parameters = lhs.get_parameters();
      const auto rhs_parameters = rhs.get_parameters();

      for(size_t i=0; i<parameters.size(); ++i)
      {
        parameters[i] = std::make_shared<Tensor>(Tensor::ternaryExpr(*parameters[i], *lhs_parameters[i], *rhs_parameters[i], [&](double value, double lhs, double rhs){
          if(dist_mutation(engine)>=mutation_rate)
            return value;
          else
            return dist_selection(engine) == 0 ? lhs : rhs;
        }));
      }
    }
    layer->set_parameters(std::move(parameters));

    // Reset states
    auto states = layer->get_states();
    {
      for(auto& state : states)
        state = std::make_shared<Tensor>(Tensor::constant(state->size(), 0.0));
    }
    layer->set_states(std::move(states));

    return layer;
  }
}
