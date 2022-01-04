#pragma once

#include <libkann/layers/Layer.hpp>

#include <libkann/serialization/Graph.hpp>
#include <libkann/serialization/Eigen.hpp>

#include <cereal/types/vector.hpp>
#include <cereal/types/utility.hpp>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>

#include <memory>
#include <ostream>
#include <utility>

namespace kann
{
  enum Tag
  {
    TAG_DEFAULT           = 1u << 0,
    TAG_GAN_GENERATOR     = 1u << 1,
    TAG_GAN_DISCRIMINATOR = 1u << 2,
  };

  class Model
  {
  public:
    struct Node
    {
      size_t size;

      Eigen::VectorXd data;
      Eigen::RowVectorXd gradient;

      template<typename Archive>
      void serialize(Archive& archive)
      {
        archive(size, data, gradient);
      }
    };

    struct Connection
    {
      std::shared_ptr<Layer> layer;
      unsigned tag;

      Eigen::ArrayXd layerGradient;

      template<typename Archive>
      void serialize(Archive& archive)
      {
        archive(layer, layerGradient);
      }
    };

  public:
    typedef boost::adjacency_list<
      boost::vecS, boost::vecS,
      boost::bidirectionalS,
      Node, Connection
    > Graph;

    typedef boost::graph_traits<Graph>::vertex_descriptor Handle;

  public:
    struct FeedBack
    {
      Handle first, second;

      template<typename Archive>
      void serialize(Archive& archive)
      {
        archive(first, second);
      }
    };

  public:
    Model() = default;
    Model(Graph graph, Handle input, Handle output, std::vector<FeedBack> feedBacks = {});

  public:
    void write_graphviz(std::ostream& os) const;

  public:
    void feedForward(Eigen::VectorXd input);
    double cost(const Eigen::VectorXd& expectedOutput) const;
    void backPropagate(const Eigen::VectorXd& expectedOutput);
    void train(double learningRate, unsigned tags = TAG_DEFAULT);

  public:
    size_t inputSize()  const { return m_graph[m_input].size; }
    size_t outputSize() const { return m_graph[m_output].size; }

  public:
    const Eigen::VectorXd& input() const;
    const Eigen::VectorXd& output() const;

  public:
    static Model cross(const Model& lhs, const Model& rhs, std::default_random_engine& engine, double mutationRate);

  public:
    template<typename Archive>
    void save(Archive& archive) const
    {
      GraphOutputSerializer graphOutputSerializer(m_graph);
      archive(graphOutputSerializer);

      {
        size_t result;
        result = graphOutputSerializer.map(m_input);
        archive(result);

        result = graphOutputSerializer.map(m_output);
        archive(result);
      }

      {
        std::vector<std::pair<size_t, size_t>> result;
        std::transform(m_feedBacks.begin(), m_feedBacks.end(), std::back_inserter(result), [&](const FeedBack& feedBack){
          return std::make_pair(
            graphOutputSerializer.map(feedBack.first),
            graphOutputSerializer.map(feedBack.second)
          );
        });
        archive(result);
      }

      {
        std::vector<size_t> result;
        std::transform(m_ordering.begin(), m_ordering.end(), std::back_inserter(result), [&](const Handle& handle){
          return graphOutputSerializer.map(handle);
        });
        archive(result);
      }
    }

    template<typename Archive>
    void load(Archive& archive)
    {
      GraphInputSerializer graphInputSerializer(m_graph);
      archive(graphInputSerializer);

      {
        size_t result;
        archive(result);
        m_input = graphInputSerializer.map(result);

        archive(result);
        m_output = graphInputSerializer.map(result);
      }

      {
        std::vector<std::pair<size_t, size_t>> result;
        archive(result);
        std::transform(result.begin(), result.end(), std::back_inserter(m_feedBacks), [&](const std::pair<size_t, size_t>& p){
          return FeedBack{
            .first = graphInputSerializer.map(p.first),
            .second = graphInputSerializer.map(p.second)
          };
        });
      }

      {
        std::vector<size_t> result;
        archive(result);
        std::transform(result.begin(), result.end(), std::back_inserter(m_ordering), [&](size_t i){
          return graphInputSerializer.map(i);
        });
      }
    }

  private:
    Graph m_graph;
    Handle m_input, m_output;
    std::vector<FeedBack> m_feedBacks;

  private:
    std::vector<Handle> m_ordering;
  };

  Model buildSimpleFeedForwardModel(std::vector<std::shared_ptr<Layer>> layers, unsigned tag = TAG_DEFAULT);
  Model buildSimpleRecurrentModel(std::vector<std::shared_ptr<Layer>> layers, size_t memory, unsigned tag = TAG_DEFAULT);

  /* The returned auto encoder model is used for training purposes
   * whereas random data can be feed into the decoder model to obtain output.
   *
   * Since Model hold std::shared_ptr to Layer, training using the auto encoder
   * model could be reflected in the decoder model.
   *
   * @return [auto encoder model, decoder model] */
  std::pair<Model, Model> buildSimpleAutoEncoderModel(std::vector<std::shared_ptr<Layer>> encoderLayers, std::vector<std::shared_ptr<Layer>> decoderLayers);

  /* The returned GAN and discriminator model is used for training purpose.
   *
   * Since Model holds std::shared_ptr to Layer, the training result could be
   * reflected in the generator model.
   *
   * @return [GAN Model, generator model, discriminator model] */
  std::tuple<Model, Model, Model> buildSimpleGANModel(std::vector<std::shared_ptr<Layer>> generatorLayers, std::vector<std::shared_ptr<Layer>> discriminatorLayers);
}
