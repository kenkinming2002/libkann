#pragma once

#include <libkann/export.hpp>

#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/binary.hpp>

#include <Eigen/Eigen>

#include <memory>
#include <random>
#include <utility>
#include <optional>
#include <span>

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
    static std::unique_ptr<Layer> cross(const Layer& lhs, const Layer& rhs, std::default_random_engine& engine, double mutationRate);

  public:
    virtual void randomize(std::default_random_engine& engine);
    virtual void train(double learningRate, unsigned tags = TAG_ALL);

  public:
    unsigned tag() const { return m_tag; }
    void tag(unsigned tag) { m_tag = tag; }

  public:
    void input(Eigen::VectorXd input);
    void outputGradient(Eigen::VectorXd outputGradient);

  public:
    const Eigen::VectorXd& input() const;
    const Eigen::VectorXd& outputGradient() const;

  public:
    virtual ~Layer() = default;

  public:
    virtual std::unique_ptr<Layer> clone() const = 0;

  public:
    virtual size_t inputSize() const = 0;
    virtual size_t outputSize() const = 0;

  public:
    /* Given input, return output. Input is stored internally for use by
     * backPropgate()
     *
     * TODO: Pass input by shared_ptr
     *
     * @return output */
    virtual Eigen::VectorXd feedForward() = 0;

    /* Return input gradient. Also compute params
     * gradient, which is stored internally in unsepecified format.
     *
     * @return input gradient */
    virtual Eigen::VectorXd backPropagate() = 0;

  // We do not have reflection in c++, so that is the best we could do
  public:
    virtual std::vector<std::span<double>> params() = 0;
    virtual std::vector<std::span<const double>> params() const = 0;

    virtual std::vector<std::span<double>> paramsGradient() = 0;
    virtual std::vector<std::span<const double>> paramsGradient() const = 0;

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(m_tag);
      archive(m_input);
      archive(m_outputGradient);
    }

  private:
    unsigned m_tag = TAG_DEFAULT;

    // This two variables are not thread safe
    Eigen::VectorXd m_input;
    Eigen::VectorXd m_outputGradient;
  };
}
