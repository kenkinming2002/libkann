#include <libkann/optimizers/AdamOptimizer.hpp>

#include <libkann/Graph.hpp>

#include <libkann/operations/Impl.hpp>
#include <libkann/operations/CWise.hpp>
#include <libkann/operations/Scale.hpp>
#include <libkann/operations/Subtract.hpp>

#include <tuple>

namespace kann
{
  AdamOptimizer::AdamOptimizer(double alpha, double beta1, double beta2, double epsilon)
    : m_alpha(alpha), m_beta1(beta1), m_beta2(beta2), m_epsilon(epsilon) {}

  class SecondMomentOperation : public Operation
  {
  public:
    constexpr SecondMomentOperation(size_t size) : m_size(size) {}

  public:
    std::vector<tensor_t> process(std::vector<tensor_t> inputs) const override
    {
      return operation_process_cwise_impl<1, 1>(std::move(inputs), m_size, [](double input) {
        return std::make_tuple(input * input);
      });
    }

  private:
    size_t m_size;
  };

  // Exponential Moving Average
  class EMAOperation : public Operation
  {
  public:
    constexpr EMAOperation(size_t size, double beta) : m_size(size), m_beta(beta) {}

  public:
    std::vector<tensor_t> process(std::vector<tensor_t> inputs) const override
    {
      return operation_process_cwise_impl<2, 1>(std::move(inputs), m_size, [this](double avg, double input) {
        return std::make_tuple(m_beta * avg + (1-m_beta) * input);
      });
    }

  private:
    size_t m_size;
    double m_beta;
  };

  class IncrementOperation : public Operation
  {
  public:
    constexpr IncrementOperation() {}

  public:
    std::vector<tensor_t> process(std::vector<tensor_t> inputs) const override
    {
      return operation_process_cwise_impl<1, 1>(std::move(inputs), 1, [](double input) {
        return std::make_tuple(input + 1.0);
      });
    }
  };

  class BiasCorrectionOperation : public Operation
  {
  public:
    constexpr BiasCorrectionOperation(double beta) : m_beta(beta) {}

  public:
    std::vector<tensor_t> process(std::vector<tensor_t> inputs) const override
    {
      return operation_process_impl<2, 1>(std::move(inputs), [this](const Tensor& t, const Tensor& v)
      {
        Tensor result(v.size());
        result.asArray() = v.asArray() / (1.0 - std::pow(m_beta, t.asScalar()));
        return std::make_tuple(std::move(result));
      });
    }

  private:
    double m_beta;
  };

  class AdamOperation : public Operation
  {
  public:
    constexpr AdamOperation(double alpha, double epsilon)
      : m_alpha(alpha), m_epsilon(epsilon) {}

  public:
    std::vector<tensor_t> process(std::vector<tensor_t> inputs) const override
    {
      return operation_process_impl<2, 1>(std::move(inputs), [this](const Tensor& m_hat, const Tensor& v_hat)
      {
        Tensor result(m_hat.size());
        const double factor = m_alpha / (v_hat.asVector().norm() + m_epsilon);
        result.asArray() = factor * m_hat.asArray();
        return std::make_tuple(std::move(result));
      });
    }

  private:
    double m_alpha;
    double m_epsilon;
  };

  size_t AdamOptimizer::process(Graph& graph, Info& info, size_t size, size_t index, size_t gradient_index) const
  {
    size_t new_index = graph.add_vertex();

    size_t gradient2_index = graph.add_vertex();

    size_t m_index = graph.add_vertex();
    size_t v_index = graph.add_vertex();

    size_t m_hat_index = graph.add_vertex();
    size_t v_hat_index = graph.add_vertex();

    size_t m_new_index = graph.add_vertex();
    size_t v_new_index = graph.add_vertex();

    size_t ts_index     = graph.add_vertex();
    size_t ts_new_index = graph.add_vertex();

    size_t correction_index = graph.add_vertex();

    // Compute second moment
    operation_t second_moment_op = std::make_shared<SecondMomentOperation>(size);
    graph.add_edge(std::move(second_moment_op), {gradient_index}, {gradient2_index});

    // EMA
    operation_t ema1_op = std::make_shared<EMAOperation>(size, m_beta1);
    operation_t ema2_op = std::make_shared<EMAOperation>(size, m_beta2);
    graph.add_edge(std::move(ema1_op), {m_index, gradient_index},  {m_new_index});
    graph.add_edge(std::move(ema2_op), {v_index, gradient2_index}, {v_new_index});

    // Timestep
    operation_t increment_op = std::make_shared<IncrementOperation>();
    graph.add_edge(std::move(increment_op), {ts_index}, {ts_new_index});

    // Bias correction
    operation_t bias_correction1_op = std::make_shared<BiasCorrectionOperation>(m_beta1);
    operation_t bias_correction2_op = std::make_shared<BiasCorrectionOperation>(m_beta2);
    graph.add_edge(std::move(bias_correction1_op), {ts_new_index, m_new_index}, {m_hat_index});
    graph.add_edge(std::move(bias_correction2_op), {ts_new_index, v_new_index}, {v_hat_index});

    operation_t adam_op = std::make_shared<AdamOperation>(m_alpha, m_epsilon);
    graph.add_edge(std::move(adam_op), {m_hat_index, v_hat_index}, {correction_index});

    operation_t subtract_op = std::make_shared<SubtractOperation>(size);
    graph.add_edge(std::move(subtract_op), {index, correction_index}, {new_index});

    info.add_state(std::make_shared<const Tensor>(Tensor::constant(size, 0.0)), m_index,  m_new_index);
    info.add_state(std::make_shared<const Tensor>(Tensor::constant(size, 0.0)), v_index,  v_new_index);
    info.add_state(std::make_shared<const Tensor>(Tensor::constant(1, 0.0)),    ts_index, ts_new_index);

    return new_index;
  }
}

