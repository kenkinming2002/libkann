#pragma once

#include <cstdint>

namespace kann
{
  struct ActivationFunction
  {
  public:
    enum class Type : uint8_t
    {
      IDENTITY,
      SIGMOID,
      TANH
    };

  public:
    ActivationFunction() = default;
    explicit ActivationFunction(Type type) : type(type) {}

  public:
    double normal(double val) const;
    double derivative(double val) const;

  public:
    template<typename Archive>
    void serialize(Archive& archive)
    {
      archive(type);
    }

  public:
    Type type;
  };
}
