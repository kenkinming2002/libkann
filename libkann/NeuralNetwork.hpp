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
  LIBKANN_SYMEXPORT NeuralNetwork(const std::vector<size_t>& topology, size_t memory, PRNG& prng);

public:
  template<typename T>
  LIBKANN_SYMEXPORT void input(T i, double v)
  {
    static_assert(std::is_enum_v<T>, "Only enum is supported");
    return this->input(static_cast<size_t>(i), v);
  }
  LIBKANN_SYMEXPORT void input(size_t i, double v);

public:
  template<typename T>
  LIBKANN_SYMEXPORT double output(T i) const
  {
    static_assert(std::is_enum_v<T>, "Only enum is supported");
    return this->output(static_cast<size_t>(i));
  }
  LIBKANN_SYMEXPORT double output(size_t i) const;

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
