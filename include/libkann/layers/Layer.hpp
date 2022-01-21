#pragma once

#include <functional>
#include <libkann/export.hpp>
#include <libkann/Variable.hpp>

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

  struct Parameter
  {
  public:
    Parameter() = default;
    Parameter(size_t size)
    {
      variable = std::make_shared<const Variable>();
      value    = std::make_shared<const Tensor>(size);
      value->asArray().setZero();
    }

  public:
    Parameter(const Parameter& other)
    {
      variable = std::make_shared<const Variable>();
      value    = other.value;
    }

  public:
    std::shared_ptr<const Variable> variable;
    std::shared_ptr<const Tensor> value;

  public:
    template<typename Archive>
    void save(Archive& archive) const
    {
      archive(value);
    }

    template<typename Archive>
    void load(Archive& archive)
    {
      variable = std::make_shared<const Variable>();
      archive(value);
    }
  };

  class Layer
  {
  public:
    typedef std::vector<std::shared_ptr<const Variable>> StateVariables;

  public:
    unsigned tag() const { return m_tag; }
    void tag(unsigned tag) { m_tag = tag; }

  public:
    virtual ~Layer() = default;

  public:
    virtual std::unique_ptr<Layer> clone() const = 0;

  public:
    virtual size_t inputSize() const = 0;
    virtual size_t outputSize() const = 0;

  public:
    virtual std::vector<std::shared_ptr<const Parameter>> parameters(unsigned tags) const = 0;
    virtual std::vector<std::shared_ptr<Parameter>> parameters(unsigned tags) = 0;

    virtual std::vector<std::shared_ptr<Parameter>> makeStates() const { return {}; }

    /* @param input input variable
     * @param state old state variable
     *
     * @return [output variable, new state variable] */
    virtual std::pair<std::shared_ptr<const Variable>, StateVariables> operator()(std::shared_ptr<const Variable> input, StateVariables state = {}) const = 0;

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(m_tag);
    }

  private:
    unsigned m_tag = TAG_DEFAULT;
  };

  void randomize(Layer& layer, std::default_random_engine& engine);
  std::unique_ptr<Layer> cross(const Layer& lhs, const Layer& rhs, std::default_random_engine& engine, double mutationRate);

}
