#include <libkann/optimizers/AdamOptimizer.hpp>

#include <libkann/operations/MultiplyOperation.hpp>
#include <libkann/operations/SubtractOperation.hpp>
#include <libkann/operations/CWiseOperation.hpp>

#include <type_traits>
#include <utility>

namespace kann
{
  class SecondMomentOperation : public CWiseOperation<SecondMomentOperation, 1>
  {
  public:
    double forward(double a) const
    {
      return a * a;
    }

    double backward(...) const
    {
      assert(false && "Unimplemented");
    }
  };

  // Exponential Moving Average
  class EMAOperation : public CWiseOperation<EMAOperation, 2>
  {
  public:
    constexpr EMAOperation(double beta) : m_beta(beta) {}

  public:
    double forward(double avg, double a) const
    {
      return m_beta * avg + (1-m_beta) * a;
    }

    double backward(...) const
    {
      assert(false && "Unimplemented");
    }

  private:
    double m_beta;
  };

  class IncrementOperation : public CWiseOperation<IncrementOperation, 1>
  {
  public:
    double forward(double a) const
    {
      return a+1;
    }

    double backward(...) const
    {
      assert(false && "Unimplemented");
    }
  };

  class BiasCorrectionOperation : public BinaryOperation
  {
  public:
    constexpr BiasCorrectionOperation(double beta) : m_beta(beta) {}

  public:
    Tensor processImpl(const Tensor& t, const Tensor& v) const override
    {
      Tensor result(v.size());
      result.asArray() = v.asArray() / (1.0 - std::pow(m_beta, t.asScalar()));
      return result;
    }

    std::pair<CRef<Variable>, CRef<Variable>> gradientsImpl(CRef<Variable> gradient, CRef<Variable>, CRef<Variable>) const override
    {
      assert(false && "Unimplemented");
    }

  private:
    double m_beta;
  };

  class AdamOperation : public BinaryOperation
  {
  public:
    constexpr AdamOperation(double alpha, double epsilon)
      : m_alpha(alpha), m_epsilon(epsilon) {}

  public:
    Tensor processImpl(const Tensor& m_hat, const Tensor& v_hat) const override
    {
      const double factor = m_alpha / (v_hat.asVector().norm() + m_epsilon);

      Tensor result(m_hat.size());
      result.asArray() = factor * m_hat.asArray();
      return result;
    }

    std::pair<CRef<Variable>, CRef<Variable>> gradientsImpl(CRef<Variable> gradient, CRef<Variable>, CRef<Variable>) const override
    {
      assert(false && "Unimplemented");
    }

  private:
    double m_alpha;
    double m_epsilon;
  };

  AdamOptimizer::AdamOptimizer(double alpha, double beta1, double beta2, double epsilon)
    : m_alpha(alpha),
      m_beta1(beta1), m_beta2(beta2),
      m_epsilon(epsilon) {}

  Optimizer::ProcessOutput AdamOptimizer::process(ProcessInput input) const
  {
    ProcessOutput output;

    // First and second moment
    auto g  = input.gradient;
    auto g2 = std::make_shared<const Variable>(std::vector{input.gradient}, std::make_shared<SecondMomentOperation>());

    // EMA calculation
    auto m     = std::make_shared<const Variable>();
    auto v     = std::make_shared<const Variable>();
    auto m_new = std::make_shared<const Variable>(std::vector{m, g},  std::make_shared<EMAOperation>(m_beta1));
    auto v_new = std::make_shared<const Variable>(std::vector{v, g2}, std::make_shared<EMAOperation>(m_beta2));

    // Timestep
    auto ts     = std::make_shared<const Variable>();
    auto ts_new = std::make_shared<const Variable>(std::vector{ts}, std::make_shared<IncrementOperation>());

    // Bias correction
    auto m_hat = std::make_shared<const Variable>(std::vector{ts_new, m_new}, std::make_shared<BiasCorrectionOperation>(m_beta1));
    auto v_hat = std::make_shared<const Variable>(std::vector{ts_new, v_new}, std::make_shared<BiasCorrectionOperation>(m_beta2));

    auto correction = std::make_shared<const Variable>(std::vector{m_hat, v_hat}, std::make_shared<AdamOperation>(m_alpha, m_epsilon));

    output.parameter = std::make_shared<const Variable>(
      std::vector{input.parameter, correction},
      std::make_shared<SubtractOperation>()
    );

    output.initial_states = {
      std::make_shared<const Tensor>(Tensor::constant(input.size, 0.0)),
      std::make_shared<const Tensor>(Tensor::constant(input.size, 0.0)),
      std::make_shared<const Tensor>(Tensor::constant(1, 0.0))
    };
    output.input_states = {m, v, ts};
    output.output_states = {m_new, v_new, ts_new};

    return output;
  }
}

