#pragma once

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
    Model(std::shared_ptr<const Layer> layer);

  public:
    void randomize(std::default_random_engine& engine);

  public:
    std::shared_ptr<const Tensor> predict(std::shared_ptr<const Tensor> input);

    // FIXME: better formatting
    std::pair<std::vector<std::shared_ptr<const Tensor>>, std::vector<double>> optimize(
      std::shared_ptr<const Optimizer> optimizer, Tag tag,
      std::vector<std::shared_ptr<const Tensor>> inputs,
      std::vector<std::shared_ptr<const Tensor>> expectedOutputs);

  public:
    friend std::shared_ptr<Model> cross(const Model& lhs, const Model& rhs, std::default_random_engine& engine, double mutationRate);

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(m_layer);
      archive(m_parametersMap);
      archive(m_statesMap);
    }

  private:
    std::shared_ptr<const Layer> m_layer;

    struct ParameterInfo
    {
      TRef value;
      size_t size;
      double mean;
      double stddev;

      template<typename Archive>
      void serialize(Archive& archive)
      {
        archive(value, size, mean, stddev);
      }
    };

    struct StateInfo
    {
      TRef value;
      size_t size;

      template<typename Archive>
      void serialize(Archive& archive)
      {
        archive(value, size);
      }
    };

    std::unordered_map<QualifiedName, ParameterInfo> m_parametersMap;
    std::unordered_map<QualifiedName, StateInfo>     m_statesMap;

  // Internal transient states
  private:
    std::unique_ptr<Executor> m_predictExecutor;

    struct OptimizeConfig
    {
      std::shared_ptr<const Optimizer> optimizer;
      Tag tag;
      size_t batchSize;

      auto operator<=>(const OptimizeConfig&) const = default;
    };

    struct OptimizeState
    {
      std::unique_ptr<Executor> executor;
      TMap map;
    };

  private:
    OptimizeState& optimizeState(std::shared_ptr<const Optimizer> optimizer, Tag tag, size_t batchSize);

  private:
    std::map<OptimizeConfig, OptimizeState> m_optimizeStates;
  };

  std::shared_ptr<Model> cross(const Model& lhs, const Model& rhs, std::default_random_engine& engine, double mutationRate);
}
