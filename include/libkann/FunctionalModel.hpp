#pragma once

#include <libkann/layers/Layer.hpp>
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
  class FunctionalModel : public Layer
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
    void write_graphviz(std::ostream& os) const;

  public:
    static std::unique_ptr<FunctionalModel> cross(const FunctionalModel& lhs, const FunctionalModel& rhs, std::default_random_engine& engine, double mutationRate);

  public:
    void randomize(std::default_random_engine& engine);
    void train(double learningRate, unsigned tags = TAG_ALL);

  public:
    std::unique_ptr<Layer> clone() const override;

  public:
    size_t inputSize()  const override { return m_graph[m_input_vertex_descriptor].size; }
    size_t outputSize() const override { return m_graph[m_output_vertex_descriptor].size; }

  public:
    Eigen::VectorXd feedForward() override;
    Eigen::VectorXd backPropagate() override;

  public:
    std::vector<std::span<double>> params() override;
    std::vector<std::span<const double>> params() const override;

    std::vector<std::span<double>> paramsGradient() override;
    std::vector<std::span<const double>> paramsGradient() const override;

  private:
    struct Node
    {
      // TODO: Consider using a shared_ptr
      size_t size;
      Eigen::VectorXd data;
      Eigen::VectorXd gradient;

      template<typename Archive>
      void serialize(Archive& archive)
      {
        archive(size, data, gradient);
      }
    };

    struct Connection
    {
      std::shared_ptr<Layer> layer;

      template<typename Archive>
      void serialize(Archive& archive)
      {
        archive(layer);
      }
    };

    typedef boost::adjacency_list<
      boost::vecS, boost::vecS,
      boost::bidirectionalS,
      Node, Connection
    > graph_type;
    typedef typename boost::graph_traits<graph_type>::vertex_descriptor vertex_descriptor_type;

    struct FeedBackVertexDescriptors
    {
      vertex_descriptor_type input_vertex_descriptor, output_vertex_descriptor;
    };

  public:
    template<typename Archive>
    void save(Archive& archive) const
    {
      archive(cereal::base_class<Layer>(this));

      GraphOutputSerializer graphOutputSerializer(m_graph);
      archive(graphOutputSerializer);

      size_t input, output;
      input  = graphOutputSerializer.map(m_input_vertex_descriptor);
      output = graphOutputSerializer.map(m_output_vertex_descriptor);
      archive(input, output);

      std::vector<std::pair<size_t, size_t>> m_feedBacks_indices;
      std::transform(m_feedBacks_vertex_descriptors.begin(), m_feedBacks_vertex_descriptors.end(), std::back_inserter(m_feedBacks_indices), [&graphOutputSerializer](const auto& feedBack_vertex_descriptor){
        return std::make_pair(
          graphOutputSerializer.map(feedBack_vertex_descriptor.input_vertex_descriptor),
          graphOutputSerializer.map(feedBack_vertex_descriptor.output_vertex_descriptor)
        );
      });
      archive(m_feedBacks_indices);

      std::vector<size_t> m_ordering_indices;
      std::transform(m_ordering.begin(), m_ordering.end(), std::back_inserter(m_ordering_indices), [&graphOutputSerializer](vertex_descriptor_type vertex_descriptor){
        return graphOutputSerializer.map(vertex_descriptor);
      });
      archive(m_ordering_indices);
    }

    template<typename Archive>
    void load(Archive& archive)
    {
      archive(cereal::base_class<Layer>(this));

      GraphInputSerializer graphInputSerializer(m_graph);
      archive(graphInputSerializer);

      size_t input, output;
      archive(input, output);
      m_input_vertex_descriptor  = graphInputSerializer.map(input);
      m_output_vertex_descriptor = graphInputSerializer.map(output);

      std::vector<std::pair<size_t, size_t>> m_feedBacks_indices;
      archive(m_feedBacks_indices);
      std::transform(m_feedBacks_indices.begin(), m_feedBacks_indices.end(), std::back_inserter(m_feedBacks_vertex_descriptors), [&graphInputSerializer](const auto& feedBack_indices){
        return FeedBackVertexDescriptors{
          .input_vertex_descriptor  = graphInputSerializer.map(feedBack_indices.first),
          .output_vertex_descriptor = graphInputSerializer.map(feedBack_indices.second)
        };
      });

      std::vector<size_t> m_ordering_indices;
      archive(m_ordering_indices);
      std::transform(m_ordering_indices.begin(), m_ordering_indices.end(), std::back_inserter(m_ordering), [&graphInputSerializer](size_t index){
        return graphInputSerializer.map(index);
      });
    }

  private:
    graph_type m_graph;

    vertex_descriptor_type m_input_vertex_descriptor, m_output_vertex_descriptor;
    std::vector<FeedBackVertexDescriptors> m_feedBacks_vertex_descriptors;
    std::vector<vertex_descriptor_type> m_ordering;
  };

  std::shared_ptr<FunctionalModel> buildSimpleFeedForwardModel(std::vector<std::shared_ptr<Layer>> layers, unsigned tag = TAG_DEFAULT);
  std::shared_ptr<FunctionalModel> buildSimpleRecurrentModel(std::vector<std::shared_ptr<Layer>> layers, size_t memory, unsigned tag = TAG_DEFAULT);

  /* The returned auto encoder model is used for training purposes
   * whereas random data can be feed into the decoder model to obtain output.
   *
   * Since Model hold std::shared_ptr to Layer, training using the auto encoder
   * model could be reflected in the decoder model.
   *
   * @return [auto encoder model, decoder model] */
  std::pair<std::shared_ptr<FunctionalModel>, std::shared_ptr<FunctionalModel>> buildSimpleAutoEncoderModel(std::vector<std::shared_ptr<Layer>> encoderLayers, std::vector<std::shared_ptr<Layer>> decoderLayers);

  /* The returned GAN and discriminator model is used for training purpose.
   *
   * Since Model holds std::shared_ptr to Layer, the training result could be
   * reflected in the generator model.
   *
   * @return [GAN Model, generator model, discriminator model] */
  std::tuple<std::shared_ptr<FunctionalModel>, std::shared_ptr<FunctionalModel>, std::shared_ptr<FunctionalModel>> buildSimpleGANModel(std::vector<std::shared_ptr<Layer>> generatorLayers, std::vector<std::shared_ptr<Layer>> discriminatorLayers);
}

namespace cereal
{
  template<typename Archive>
  struct specialize<Archive, kann::FunctionalModel, cereal::specialization::member_load_save> {};
}
