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
    Model(CRef<Layer> layer);

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
      archive(m_parametersMap);
      archive(m_statesMap);
    }

  private:
    CRef<Layer> m_layer;

    struct ParameterInfo
    {
      CRef<Tensor> value;
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
      CRef<Tensor> value;
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
      CRef<Optimizer> optimizer;
      Tag tag;
      size_t batchSize;

      auto operator<=>(const OptimizeConfig&) const = default;
    };

    struct OptimizeState
    {
      std::unique_ptr<Executor> executor;
      Map<Tensor> map;
    };

  private:
    OptimizeState& optimizeState(CRef<Optimizer> optimizer, Tag tag, size_t batchSize);

  private:
    std::map<OptimizeConfig, OptimizeState> m_optimizeStates;
  };

  Ref<Model> cross(const Model& lhs, const Model& rhs, std::default_random_engine& engine, double mutationRate);
}
