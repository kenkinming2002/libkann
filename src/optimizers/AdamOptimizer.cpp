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

    VariablePair gradientsImpl(VariableHandle gradient, VariableHandle, VariableHandle) const override
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

    VariablePair gradientsImpl(VariableHandle gradient, VariableHandle, VariableHandle) const override
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

  void AdamOptimizer::process(Context& context) const
  {
    // First and second moment
    auto g  = context.gradient;
    auto g2 = std::make_shared<const Variable>(std::vector{context.gradient}, std::make_shared<SecondMomentOperation>());

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

    context.outputParameter = std::make_shared<const Variable>(
      std::vector{context.inputParameter, correction},
      std::make_shared<SubtractOperation>()
    );

    auto m_name  = QualifiedName{
      .scope = context.qualifiedName.toScope(),
      .name = "m",
      .size = context.qualifiedName.size
    };
    auto v_name  = QualifiedName{
      .scope = context.qualifiedName.toScope(),
      .name = "v",
      .size = context.qualifiedName.size
    };
    auto ts_name = QualifiedName{
      .scope = context.qualifiedName.toScope(),
      .name = "ts",
      .size = 1
    };

    context.inputState.emplace(m_name,  m);
    context.inputState.emplace(v_name,  v);
    context.inputState.emplace(ts_name, ts);

    context.outputState.emplace(m_name,   m_new);
    context.outputState.emplace(v_name,   v_new);
    context.outputState.emplace(ts_name,  ts_new);

    context.initialState.emplace(m_name, std::make_shared<const Tensor>(Tensor::constant(context.qualifiedName.size, 0.0)));
    context.initialState.emplace(v_name, std::make_shared<const Tensor>(Tensor::constant(context.qualifiedName.size, 0.0)));
    context.initialState.emplace(ts_name, std::make_shared<const Tensor>(Tensor::constant(1, 0.0)));
  }
}

