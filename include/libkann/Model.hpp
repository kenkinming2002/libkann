#pragma once

#include <libkann/layers/Layer.hpp>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>

#include <memory>
#include <ostream>
#include <utility>

namespace kann
{
  class Model
  {
  private:
    struct Node
    {
      size_t size;

      Eigen::VectorXd data;
      Eigen::RowVectorXd gradient;
    };

    struct Connection
    {
      std::shared_ptr<Layer> layer;

      // Currently unsupported
      size_t inputOffset;
      size_t outputOffset;

      Eigen::ArrayXd layerGradient;
    };

    typedef boost::adjacency_list<
      boost::vecS, boost::vecS,
      boost::bidirectionalS,
      Node, Connection
    > Graph;

  public:
    typedef boost::graph_traits<Graph>::vertex_descriptor Handle;

  public:
    Handle addNode(size_t size);
    void addConnection(Handle parent, Handle child, std::shared_ptr<Layer> layer, size_t inputOffset = 0, size_t outputOffset = 0);
    void build(Handle input, Handle output);

  public:
    void write_graphviz(std::ostream& os) const;

  public:
    void feedForward(Eigen::VectorXd input);
    void backPropagate(const Eigen::VectorXd& expectedOutput);
    void train(double learningRate);

  public:
    Eigen::VectorXd output() const;

  private:
    Graph m_graph;

  private:
    Handle m_input, m_output;
    std::vector<Handle> m_ordering;
  };

  Model buildSimpleFeedForwardModel(std::vector<std::shared_ptr<Layer>> layers);

  /* The returned auto encoder model is used for training purposes
   * whereas random data can be feed into the decoder model to obtain output.
   *
   * Since Model hold std::shared_ptr to Layer, training using the auto encoder
   * model could be reflected in the decoder model.
   *
   * @return [auto encoder model, decoder model] */
  std::pair<Model, Model> buildSimpleAutoEncoderModel(std::vector<std::shared_ptr<Layer>> encoderLayers, std::vector<std::shared_ptr<Layer>> decoderLayers);
}
