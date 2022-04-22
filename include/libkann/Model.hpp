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
    std::shared_ptr<const Layer> layer() const { return m_layer; }

  public:
    void compile(size_t batch_size, std::shared_ptr<const Optimizer> optimizer, std::vector<Tag> tags);
    void randomize(std::default_random_engine& engine);

  public:
    std::shared_ptr<const Tensor> predict(std::shared_ptr<const Tensor> input);

    std::pair<std::vector<std::shared_ptr<const Tensor>>, std::vector<double>> optimize(
      std::vector<std::shared_ptr<const Tensor>> inputs,
      std::vector<std::shared_ptr<const Tensor>> expected_outputs,
      Tag tag);

  public:
    template<typename Archive>
    void save(Archive& archive) const
    {
      archive(m_layer);
      archive(m_batch_size, m_optimizer, m_tags);
    }

    template<typename Archive>
    void load(Archive& archive)
    {
      archive(m_layer);
      archive(m_batch_size, m_optimizer);
      compile(m_batch_size, m_optimizer, m_tags);
    }

  private:
    std::shared_ptr<Layer> m_layer;

  public:
    size_t m_batch_size;
    std::shared_ptr<const Optimizer> m_optimizer;
    std::vector<Tag> m_tags;

  private:
    std::unique_ptr<Executor> m_executor;

    // Could be shared
    // Ideally, I don't even want to need to explicitly cache Graph
    std::shared_ptr<const Graph> m_predict_graph;
    std::unordered_map<Tag, std::shared_ptr<const Graph>> m_optimize_graphs;
    std::unordered_map<Tag, std::vector<std::shared_ptr<const Tensor>>> m_optimize_states;
  };
}
