#pragma once

#include <libkann/Types.hpp>
#include <libkann/Tensor.hpp>
#include <libkann/Layer.hpp>
#include <libkann/Executor.hpp>
#include <libkann/Optimizer.hpp>

#include <cereal/types/unordered_map.hpp>

#include <random>
#include <vector>
#include <memory>

namespace kann
{
  class Model
  {
  public:
    Model() = default;
    Model(std::shared_ptr<Layer> layer);

  public:
    void randomize(std::default_random_engine& engine);

  public:
    CRef<Tensor> predict(CRef<Tensor> input);

    // FIXME: better formatting
    std::pair<std::vector<CRef<Tensor>>, std::vector<double>> optimize(
      CRef<Optimizer> optimizer, Tag tag,
      std::vector<CRef<Tensor>> inputs,
      std::vector<CRef<Tensor>> expectedOutputs);

  public:
    friend Ref<Model> cross(const Model& lhs, const Model& rhs, std::default_random_engine& engine, double mutationRate);

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(m_layer);
    }

  private:
    std::shared_ptr<Layer> m_layer;

  // Internal transient states
  private:
    struct OptimizeConfig
    {
      CRef<Optimizer> optimizer;
      Tag tag;
      size_t batchSize;

      auto operator<=>(const OptimizeConfig&) const = default;
    };

    struct OptimizeState
    {
      std::unique_ptr<Executor> executor;
      std::vector<CRef<Tensor>> values;
    };

    std::unique_ptr<Executor> m_predictExecutor;
    std::map<OptimizeConfig, OptimizeState> m_optimizeStates;

  private:
    Executor& predictExecutor();
    OptimizeState& optimizeState(CRef<Optimizer> optimizer, Tag tag, size_t batchSize);
  };

  Ref<Model> cross(const Model& lhs, const Model& rhs, std::default_random_engine& engine, double mutationRate);
}
