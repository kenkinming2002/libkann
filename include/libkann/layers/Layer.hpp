#pragma once

#include <libkann/export.hpp>

#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/binary.hpp>

#include <Eigen/Eigen>

#include <memory>
#include <random>
#include <utility>
#include <concepts>

namespace kann
{
  enum Tag
  {
    TAG_DEFAULT           = 1u << 0,
    TAG_ENCODDER          = 1u << 1,
    TAG_DECODDER          = 1u << 2,
    TAG_GAN_GENERATOR     = 1u << 3,
    TAG_GAN_DISCRIMINATOR = 1u << 4,
    TAG_ALL = 0xFFFFFFFF
  };

  class Layer
  {
  public:
    Layer() = default;
    Layer(size_t paramsCount);

  public:
    virtual ~Layer() = default;

  public:
    virtual std::unique_ptr<Layer> clone() const = 0;

  public:
    virtual size_t inputSize() const = 0;
    virtual size_t outputSize() const = 0;

  public:
    virtual void feedForward(const Eigen::VectorXd& input, Eigen::VectorXd& output) const = 0;
    virtual void backPropagate(const Eigen::VectorXd& input, const Eigen::RowVectorXd& outputGradient, Eigen::RowVectorXd& inputGradient, Eigen::ArrayXd& layerGradient) const = 0;

  public:
    static std::unique_ptr<Layer> cross(const Layer& lhs, const Layer& rhs, std::default_random_engine& engine, double mutationRate);

    void randomize(std::default_random_engine& engine);
    void train(double learningRate, const Eigen::ArrayXd& layerGradient);

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(m_params);
    }

  public:
    unsigned tag() const { return m_tag; }
    void tag(unsigned tag) { m_tag = tag; }

  protected:
    const Eigen::ArrayXd& params() const { return m_params; }

  private:
    unsigned m_tag = TAG_DEFAULT;
    Eigen::ArrayXd m_params;
  };
}
