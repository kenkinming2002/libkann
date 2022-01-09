#pragma once

#include <libkann/layers/Layer.hpp>
#include <libkann/Variable.hpp>

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
  class Model
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
    Model() = default;
    Model(std::shared_ptr<const Variable> input, std::shared_ptr<const Variable> output,
        std::vector<FeedBack> feedBacks = {});

  private:
    void initialize();

  public:
    void write_graphviz(std::ostream& os) const;

  public:
    void feedForward(Eigen::VectorXd input);
    double cost(const Eigen::VectorXd& expectedOutput) const;
    void backPropagate(const Eigen::VectorXd& expectedOutput);
    void train(double learningRate, unsigned tags = TAG_ALL);

  public:
    size_t inputSize()  const { return m_input->size; }
    size_t outputSize() const { return m_output->size; }

  public:
    const Eigen::VectorXd& input()  const { return m_graph[m_inputHandle].data; }
    const Eigen::VectorXd& output() const { return m_graph[m_outputHandle].data; }

  public:
    static Model cross(const Model& lhs, const Model& rhs, std::default_random_engine& engine, double mutationRate);

  public:
    template<typename Archive>
    void save(Archive& archive) const
    {
      archive(m_input, m_output, m_feedBacks);
    }

    template<typename Archive>
    void load(Archive& archive)
    {
      archive(m_input, m_output, m_feedBacks);
      initialize();
    }

  // Externally visible representation
  private:
    std::shared_ptr<const Variable> m_input, m_output;
    std::vector<FeedBack> m_feedBacks;

  // Internal representation, could be reconstructed
  private:
    struct Node
    {
      std::shared_ptr<const Variable> variable;
      Eigen::VectorXd data;
      Eigen::RowVectorXd gradient;
    };

    struct Connection
    {
      std::shared_ptr<Layer> layer;
      Eigen::ArrayXd layerGradient;
    };

  private:
    typedef boost::adjacency_list<
      boost::vecS, boost::vecS,
      boost::bidirectionalS,
      Node, Connection
    > Graph;

    typedef boost::graph_traits<Graph>::vertex_descriptor Handle;

    struct FeedBackHandle
    {
      Handle input, output;
    };

  private:
    Graph m_graph;
    Handle m_inputHandle, m_outputHandle;
    std::vector<FeedBackHandle> m_feedBackHandles;
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
