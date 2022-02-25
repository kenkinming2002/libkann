#pragma once

#include <libkann/operations/CWiseProductOperation.hpp>
#include <libkann/Operation.hpp>

#include <type_traits>
#include <utility>

namespace kann
{
  /* CRTP Black magic */
  template<typename Derived, size_t Count>
  class CWiseOperation : public Operation
  {
  public:
    CRef<Tensor> process(std::vector<const Tensor*> inputs) const override
    {
      return _process(std::move(inputs), std::make_index_sequence<Count>{});
    }

    std::vector<CRef<Variable>> gradients(CRef<Variable> gradient, std::vector<CRef<Variable>> inputs) const override
    {
      return _gradients(std::move(gradient), std::move(inputs), std::make_index_sequence<Count>{});
    }

  private:
    const Derived& derived() const { return static_cast<const Derived&>(*this); }

  private:
    template<size_t... Ints>
    CRef<Tensor> _process(std::vector<const Tensor*> inputs, std::index_sequence<Ints...>) const
    {
      const size_t size = inputs.front()->size();
      auto result = std::make_shared<Tensor>(size);
      for(size_t i=0; i<size; ++i)
        result->asArray()(i) = derived().forward(inputs[Ints]->asArray()(i)...);

      return result;
    }

  private:
    // I need to generate required gradient operation
    class GradientOperation : public Operation
    {
    public:
      constexpr GradientOperation(size_t index, Derived op) : m_index(index), m_op(op) {}

    public:
      CRef<Tensor> process(std::vector<const Tensor*> inputs) const override
      {
        return _process(std::move(inputs), std::make_index_sequence<Count+1>{});
      }

      std::vector<CRef<Variable>> gradients(CRef<Variable> gradient, std::vector<CRef<Variable>> inputs) const override
      {
        assert(false && "Unimplemented");
      }

    private:
      template<size_t... Ints>
      CRef<Tensor> _process(std::vector<const Tensor*> inputs, std::index_sequence<Ints...>) const
      {
        const size_t size = inputs.front()->size();
        auto result = std::make_shared<Tensor>(size);
        for(size_t i=0; i<size; ++i)
          result->asArray()(i) = m_op.derived().backward(m_index, inputs[Ints]->asArray()(i)...);

        return result;
      }

    private:
      size_t m_index;
      Derived m_op;
    };

    template<size_t... Ints>
    std::vector<CRef<Variable>> _gradients(CRef<Variable> gradient, std::vector<CRef<Variable>> inputs, std::index_sequence<Ints...> seq) const
    {
      std::vector<CRef<Variable>> realInput;
      realInput.push_back(std::move(gradient));
      realInput.insert(realInput.end(),
        std::move_iterator(inputs.begin()),
        std::move_iterator(inputs.end())
      );

      std::vector<CRef<Variable>> result;
      result.reserve(Count);
      auto f = [&](size_t i)
      {
        auto variable = std::make_shared<const Variable>(realInput, std::make_shared<GradientOperation>(i, derived()));
        result.push_back(std::move(variable));
      };
      (f(Ints), ...);
      return result;
    }
  };
}
