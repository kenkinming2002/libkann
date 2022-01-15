#pragma once

#include <libkann/Variable.hpp>

namespace kann
{
  class Function
  {
  public:
    virtual std::shared_ptr<const Variable> operator()(std::vector<std::shared_ptr<const Variable>> inputs) const = 0;
    virtual ~Function() = default;
  };

  class UnaryFunction : public Function
  {
  public:
    std::shared_ptr<const Variable> operator()(std::vector<std::shared_ptr<const Variable>> inputs) const override final
    {
      assert(inputs.size() == 1);
      return this->impl(std::move(inputs[0]));
    }

  protected:
    virtual std::shared_ptr<const Variable> impl(std::shared_ptr<const Variable>) const = 0;
  };

  class BinaryFunction : public Function
  {
  public:
    std::shared_ptr<const Variable> operator()(std::vector<std::shared_ptr<const Variable>> inputs) const override final
    {
      assert(inputs.size() == 2);
      return this->impl(std::move(inputs[0]), std::move(inputs[1]));
    }

  protected:
    virtual std::shared_ptr<const Variable> impl(std::shared_ptr<const Variable>, std::shared_ptr<const Variable>) const = 0;
  };

  class TernaryFunction : public Function
  {
  public:
    std::shared_ptr<const Variable> operator()(std::vector<std::shared_ptr<const Variable>> inputs) const override final
    {
      assert(inputs.size() == 3);
      return this->impl(std::move(inputs[0]), std::move(inputs[1]), std::move(inputs[2]));
    }

  protected:
    virtual std::shared_ptr<const Variable> impl(std::shared_ptr<const Variable>, std::shared_ptr<const Variable>, std::shared_ptr<const Variable>) const = 0;
  };
}
