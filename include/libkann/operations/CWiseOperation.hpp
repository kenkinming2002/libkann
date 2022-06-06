#pragma once

#include <libkann/Utility.hpp>
#include <libkann/Operation.hpp>

#include <utility>
#include <assert.h>

namespace kann
{
  template<typename Derived, size_t M, size_t N>
  class CWiseOperation : public OperationImpl<CWiseOperation<Derived, M, N>, M, N>
  {
  public:
    using Base = OperationImpl<CWiseOperation<Derived, M, N>, M, N>;

    using typename Base::inputs_t;
    using typename Base::outputs_t;

    using cwise_inputs_t  = std::array<double, M>;
    using cwise_outputs_t = std::array<double, N>;

  public:
    cwise_outputs_t forward(cwise_inputs_t inputs) const = delete;
    cwise_inputs_t backward(cwise_inputs_t inputs, cwise_outputs_t output_gradients) const = delete;

  public:
    constexpr CWiseOperation(size_t size) : m_size(size) {}

  public:
    const Derived& derived() const { return static_cast<const Derived&>(*this); }

  public:
    outputs_t process_impl(inputs_t inputs) const
    {
      assert(ranges::all_of(inputs, [this](const tensor_t& input) { return input->size() == m_size; } ));

      std::array<Tensor, N> outputs;
      for(Tensor& output : outputs)
        output = Tensor(m_size);

      for(size_t i=0; i<m_size; ++i)
      {
        cwise_inputs_t cwise_inputs;
        for(size_t j=0; j<M; ++j)
          cwise_inputs[j] = inputs[j]->asArray()(i);

        cwise_outputs_t cwise_outputs = derived().forward(cwise_inputs);
        for(size_t j=0; j<N; ++j)
          outputs[j].asArray()(i) = cwise_outputs[j];
      }

      return outputs;
    }

  private:
    size_t m_size;

  public:
    class GradientOperation : public OperationImpl<GradientOperation, M+N, M>
    {
    public:
      using Base = OperationImpl<GradientOperation, M+N, M>;

      using typename Base::inputs_t;
      using typename Base::outputs_t;

      using cwise_inputs_t  = std::array<double, M>;
      using cwise_outputs_t = std::array<double, N>;

    public:
      constexpr GradientOperation(Derived derived) : m_derived(derived) {}

    public:
      outputs_t process_impl(inputs_t inputs) const
      {
        const size_t size = static_cast<const CWiseOperation&>(m_derived).m_size;

        assert(ranges::all_of(inputs, [size](const tensor_t& input) { return input->size() == size; } ));

        std::array<Tensor, M> input_gradients;
        for(Tensor& input_gradient : input_gradients)
          input_gradient = Tensor(size);

        for(size_t i=0; i<size; ++i)
        {
          cwise_inputs_t cwise_inputs;
          for(size_t j=0; j<M; ++j)
            cwise_inputs[j] = inputs[j]->asArray()(i);

          cwise_outputs_t cwise_output_gradients;
          for(size_t j=0; j<N; ++j)
            cwise_output_gradients[j] = inputs[M+j]->asArray()(i);

          cwise_inputs_t cwise_input_gradients = m_derived.backward(cwise_inputs, cwise_output_gradients);
          for(size_t j=0; j<M; ++j)
            input_gradients[j].asArray()(i) = cwise_input_gradients[j];
        }

        return input_gradients;
      }

    private:
      Derived m_derived;
    };

    operation_t differentiate() const override
    {
      return std::make_shared<GradientOperation>(derived());
    }

  };
}
