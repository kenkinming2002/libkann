#pragma once

#include <libkann/layers/Layer.hpp>

#include <boost/graph/adjacency_list.hpp>

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
      std::shared_ptr<Layer> layer;

      Eigen::VectorXd input;
      Eigen::RowVectorXd outputGradient;

      Eigen::ArrayXd layerGradient;
    };

  public:
    size_t add(std::shared_ptr<Layer> layer);
    void connect(size_t parentID, size_t childID);
    void build(size_t inputID, size_t outputID);

  public:
    void write_graphviz(std::ostream& os) const;

  public:
    Eigen::VectorXd feedForward(Eigen::VectorXd input);
    void backPropagate(const Eigen::VectorXd& output, const Eigen::VectorXd& expectedOutput);
    void train(double learningRate);

  private:
    std::vector<Node> m_nodes;

  private:
    size_t m_inputID, m_outputID;

  private:
    struct FeedbackBuffer
    {
      size_t inputID;
      Eigen::VectorXd buffer;
    };

    // Index by outputID
    std::map<size_t, FeedbackBuffer> m_feedbackBuffers;

  private:
    boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS> m_graph;
    std::vector<size_t> m_ordering;
  };

  Model buildSimpleFeedForwardModel(std::vector<std::shared_ptr<Layer>> layers);

  /* The returend auto encoder model is used for training purposes
   * whereas random data can be feed into the decoder model to obtain output.
   *
   * Since Model hold std::shared_ptr to Layer, training using the auto encoder
   * model could be reflected in the decoder model.
   *
   * @return [auto encoder model, decoder model] */
  std::pair<Model, Model> buildSimpleAutoEncoderModel(std::vector<std::shared_ptr<Layer>> encoderLayers, std::vector<std::shared_ptr<Layer>> decoderLayers);
}
