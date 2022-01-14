#pragma once

#include <libkann/Model.hpp>
#include <libkann/Variable.hpp>

#include <libkann/serialization/Graph.hpp>
#include <libkann/serialization/Eigen.hpp>

#include <cereal/specialize.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/utility.hpp>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>

#include <memory>
#include <ostream>
#include <utility>

namespace kann
{
  class FunctionalModel : public Model
  {
  public:
    struct FeedBack
    {
      std::shared_ptr<const Variable> input, output;

      template<typename Archive>
      void serialize(Archive& archive)
      {
        archive(input, output);
      }
    };

  public:
    FunctionalModel() = default;
    FunctionalModel(std::shared_ptr<const Variable> input, std::shared_ptr<const Variable> output, std::vector<FeedBack> feedBacks = {});

  public:
    void write_graphviz(std::ostream& os) const override;

  public:
    std::unique_ptr<Layer> clone() const override;

  public:
    size_t inputSize()  const override { return node(m_inputNodeIndex).size; }
    size_t outputSize() const override { return node(m_inputNodeIndex).size; }

  public:
    Eigen::VectorXd feedForward() override;
    Eigen::VectorXd backPropagate() override;

  private:
    void build();

  public:
    template<typename Archive>
    void save(Archive& archive) const
    {
      archive(cereal::base_class<Model>(this));

      archive(m_nodes);
      archive(m_inputNodeIndex, m_outputNodeIndex);
      archive(m_feedBacksNodeIndices);

      archive(GraphOutputSerializer(m_graph));
    }

    template<typename Archive>
    void load(Archive& archive)
    {
      archive(cereal::base_class<Model>(this));

      archive(m_nodes);
      archive(m_inputNodeIndex, m_outputNodeIndex);
      archive(m_feedBacksNodeIndices);

      archive(GraphInputSerializer(m_graph));
      build();
    }

  // Node
  private:
    struct Node
    {
      size_t size;
      Eigen::VectorXd data;
      Eigen::VectorXd gradient;

      template<typename Archive>
      void serialize(Archive& archive)
      {
        archive(size, data, gradient);
      }
    };

  private:
    Node& node(size_t index);
    const Node& node(size_t index) const;

  private:
    std::vector<Node> m_nodes;

  private:
    size_t m_inputNodeIndex, m_outputNodeIndex;
    std::vector<std::pair<size_t, size_t>> m_feedBacksNodeIndices;

  // Graph
  private:
    struct VertexProperty
    {
      size_t nodeIndex;
      template<typename Archive>
      void serialize(Archive& archive)
      {
        archive(nodeIndex);
      }
    };

    struct EdgeProperty
    {
      size_t layerIndex;
      template<typename Archive>
      void serialize(Archive& archive)
      {
        archive(layerIndex);
      }
    };

    typedef boost::adjacency_list<
      boost::vecS, boost::vecS,
      boost::bidirectionalS,
      VertexProperty, EdgeProperty
    > graph_type;
    typedef boost::graph_traits<graph_type>::vertex_descriptor vertex_type;
    typedef boost::graph_traits<graph_type>::edge_descriptor edge_type;

  private:
    graph_type m_graph;
    std::vector<vertex_type> m_ordering;
  };
}

CEREAL_REGISTER_TYPE(kann::FunctionalModel);
CEREAL_REGISTER_POLYMORPHIC_RELATION(kann::Model, kann::FunctionalModel);

namespace cereal
{
  template<typename Archive>
  struct specialize<Archive, kann::FunctionalModel, cereal::specialization::member_load_save> {};
}
