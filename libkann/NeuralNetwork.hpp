#pragma once

#include <libkann/export.hpp>

#include <libkann/utilities/dynarray.hpp>
#include <libkann/Layer.hpp>

#include <Eigen/Eigen>

#include <ostream>
#include <vector>
#include <random>
#include <type_traits>
#include <initializer_list>

class NeuralNetwork
{
public:
  LIBKANN_SYMEXPORT NeuralNetwork(const std::vector<size_t>& topology, size_t memory);

  template<typename PRNG>
  LIBKANN_SYMEXPORT NeuralNetwork(const std::vector<size_t>& topology, PRNG& prng, size_t memory);

public:
  // TODO: Consider optimizing this
  LIBKANN_SYMEXPORT void input(std::initializer_list<double> input);

  /*
   * Returning an entire std::vector is inefficient as memory allocation is
   * invovled to convert the underlying Eigen vector to std vector, so if the
   * index is known, and only a few value is needed, the first version can be
   * used instead. A templated version taking enum class is also provided for
   * convenience.
   */
  template<typename T>
  LIBKANN_SYMEXPORT double output(T i) const
  {
    static_assert(std::is_enum_v<T>, "Only enum is supported");
    return this->output(static_cast<size_t>(i));
  }

  LIBKANN_SYMEXPORT double output(size_t i) const;
  LIBKANN_SYMEXPORT const Eigen::VectorXd& output() const;

public:
  LIBKANN_SYMEXPORT void feedForward();
  LIBKANN_SYMEXPORT void backPropagate(const std::vector<double>& input);

public:
  template<typename PRNG>
  LIBKANN_SYMEXPORT static NeuralNetwork cross(const NeuralNetwork& lhs, const NeuralNetwork& rhs, PRNG& prng, double mutationRate);

private:
  size_t m_memory;

public:
  dynarray<size_t> m_topology;
  dynarray<Layer> m_layers;
  dynarray<Eigen::MatrixXd> m_weights;

  Eigen::VectorXd m_output;
};
