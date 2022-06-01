#pragma once

#include <libkann/Operation.hpp>
#include <libkann/Variable.hpp>

#include <utility>
#include <assert.h>

namespace kann
{
  template<typename T, size_t N>
  static inline auto to_array()
  {
    return ranges::make_view_closure([](auto&& r){
      std::array<T, N> result;
      assert(ranges::size(std::forward<decltype(r)>(r)) == N);
      ranges::copy(std::forward<decltype(r)>(r), result.begin());
      return result;
    });
  }

  template<typename Derived, size_t N>
  class CWiseOperation : public OperationImpl<CWiseOperation<Derived, N>, N>
  {
  public:
    using cwise_inputs_t = std::array<double, N>;

  public:
    double forward(cwise_inputs_t inputs) = delete;

    template<size_t index>
    double backward(double gradient, cwise_inputs_t input) = delete;

  public:
    using Base = OperationImpl<CWiseOperation<Derived, N>, N>;
    using typename Base::inputs_t;
    using typename Base::variables_t;

  public:
    const Derived& derived() const { return static_cast<const Derived&>(*this); }

    template<size_t index>
    class GradientOperation : public OperationImpl<GradientOperation<index>, N+1>
    {
    public:
      constexpr GradientOperation(Derived derived) : m_derived(derived) {}

    public:
      using Base = OperationImpl<GradientOperation<index>, N+1>;
      using typename Base::inputs_t;
      using typename Base::variables_t;

    public:
      Tensor process_impl(inputs_t inputs) const
      {
        size_t size = inputs[0]->size();
        assert(ranges::all_of(inputs, [size](const Tensor* input) { return input->size() == size; } ));

        Tensor result(size);
        for(size_t i=0; i<size; ++i)
        {
          double gradient = inputs[0]->asArray()(i);
          cwise_inputs_t cwise_inputs = inputs
            | ranges::views::drop_exactly(1)
            | ranges::views::transform([i](const Tensor* input) { return input->asArray()(i); })
            | to_array<double, N>();

          result.asArray()(i) = m_derived.template backward<index>(gradient, cwise_inputs);
        }
        return result;
      }

      variables_t gradients_impl(variable_t gradient, variables_t inputs) const
      {
        assert(false && "Unimplemented");
      }

    private:
      Derived m_derived;
    };

  public:
    Tensor process_impl(inputs_t inputs) const
    {
      size_t size = inputs[0]->size();
      assert(ranges::all_of(inputs, [size](const Tensor* input) { return input->size() == size; } ));

      Tensor result(size);
      for(size_t i=0; i<size; ++i)
      {
        cwise_inputs_t cwise_inputs = inputs
          | ranges::views::transform([i](const Tensor* input) { return input->asArray()(i); })
          | to_array<double, N>();

        result.asArray()(i) = derived().forward(cwise_inputs);
      }
      return result;
    }

    template<size_t... Indices>
    variables_t _gradients_impl(variable_t gradient, variables_t inputs, std::index_sequence<Indices...>) const
    {
      std::vector<variable_t> all_inputs;
      all_inputs.reserve(N+1);
      all_inputs |= ranges::actions::push_back(gradient);
      all_inputs |= ranges::actions::push_back(inputs);

      return { Variable::apply(GradientOperation<Indices>(derived()), all_inputs) ... };
    }

    variables_t gradients_impl(variable_t gradient, variables_t inputs) const
    {
      return _gradients_impl(std::move(gradient), std::move(inputs), std::make_index_sequence<N>());
    }
  };
}
