#pragma once

#include <libkann/layers/Layer.hpp>

#include <libkann/serialization/Graph.hpp>
#include <cereal/types/vector.hpp>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>

#include <memory>
#include <ostream>
#include <utility>

namespace kann
{
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
    void backPropagate(const Eigen::VectorXd& expectedOutput);
    void train(double learningRate);

  public:
    Eigen::VectorXd output() const;

  public:
    static Model cross(const Model& lhs, const Model& rhs, std::default_random_engine& engine, double mutationRate);

  public:
    struct Pair
    {
      size_t first;
      size_t second;

      template<typename Archive>
      void serialize(Archive& archive)
      {
        archive(first, second);
      }
    };

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
        std::vector<Pair> result;
        std::transform(m_feedBacks.begin(), m_feedBacks.end(), std::back_inserter(result), [&](const FeedBack& feedBack){
          return Pair{
            .first  = graphOutputSerializer.map(feedBack.first),
            .second = graphOutputSerializer.map(feedBack.second)
          };
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
        std::vector<Pair> result;
        archive(result);
        std::transform(result.begin(), result.end(), std::back_inserter(m_feedBacks), [&](const Pair& p){
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

  Model buildSimpleFeedForwardModel(std::vector<std::shared_ptr<Layer>> layers);
  Model buildSimpleRecurrentModel(std::vector<std::shared_ptr<Layer>> layers, size_t memory);

  /* The returned auto encoder model is used for training purposes
   * whereas random data can be feed into the decoder model to obtain output.
   *
   * Since Model hold std::shared_ptr to Layer, training using the auto encoder
   * model could be reflected in the decoder model.
   *
   * @return [auto encoder model, decoder model] */
  std::pair<Model, Model> buildSimpleAutoEncoderModel(std::vector<std::shared_ptr<Layer>> encoderLayers, std::vector<std::shared_ptr<Layer>> decoderLayers);
}
