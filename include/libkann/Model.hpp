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
    template<typename Archive>
    void save(Archive& archive) const
    {
      GraphSerializer graphSerializer(m_graph);

      archive(graphSerializer);
      archive(VertexSerializer(graphSerializer, m_input));
      archive(VertexSerializer(graphSerializer, m_output));

      size_t size;
      {
        size = m_feedBacks.size();
        archive(size);
        for(auto& feedBack : m_feedBacks)
        {
          archive(VertexSerializer(graphSerializer, feedBack.first));
          archive(VertexSerializer(graphSerializer, feedBack.second));
        }
      }

      {
        size = m_ordering.size();
        archive(size);
        for(auto& handle : m_ordering)
          archive(VertexSerializer(graphSerializer, handle));
      }
    }

    template<typename Archive>
    void load(Archive& archive)
    {
      GraphSerializer graphSerializer(m_graph);

      archive(graphSerializer);
      archive(VertexSerializer(graphSerializer, m_input));
      archive(VertexSerializer(graphSerializer, m_output));

      size_t size;
      {
        archive(size);
        m_feedBacks.resize(size);
        for(auto& feedBack : m_feedBacks)
        {
          archive(VertexSerializer(graphSerializer, feedBack.first));
          archive(VertexSerializer(graphSerializer, feedBack.second));
        }
      }

      {
        archive(size);
        m_ordering.resize(size);
        for(auto& handle : m_ordering)
          archive(VertexSerializer(graphSerializer, handle));
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
