#pragma once

#include <libkann/Tensor.hpp>
#include <libkann/NewLayer.hpp>
#include <libkann/Executor.hpp>

#include <vector>
#include <memory>

namespace kann
{
  class NewModel
  {
  public:
    NewModel() = default;
    NewModel(std::shared_ptr<const NewLayer> layer);

  public:
    void randomize();

  public:
    std::shared_ptr<const Tensor> predict(std::shared_ptr<const Tensor> input);

    // FIXME: better formatting
    std::pair<std::vector<std::shared_ptr<const Tensor>>, std::vector<double>> optimize(
      double learningRate, unsigned tags,
      std::vector<std::shared_ptr<const Tensor>> inputs,
      std::vector<std::shared_ptr<const Tensor>> expectedOutputs);

  private:
    Executor& optimizeExecutor(double learningRate, unsigned tags, size_t batchSize);

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(m_layer);
      archive(m_parameters);
      archive(m_states);
    }

  private:
    std::shared_ptr<const NewLayer> m_layer;

    std::vector<std::shared_ptr<const Tensor>> m_parameters;
    std::vector<std::shared_ptr<const Tensor>> m_states;


  private:
    std::unique_ptr<Executor> m_predictExecutor;

    struct OptimizeConfig
    {
      double learningRate;
      unsigned tags;
      size_t batchSize;

      auto operator<=>(const OptimizeConfig&) const = default;
    };
    std::map<OptimizeConfig, std::unique_ptr<Executor>> m_optimizeExecutors;
  };
}
