#pragma once

#include <libkann/export.hpp>
#include <libkann/ActivationFunction.hpp>

#include <Eigen/Eigen>

#include <utility>
#include <concepts>

template<typename T>
concept isLayer = requires(T t, Eigen::VectorXd input, Eigen::RowVectorXd outputGradient) {
  { t.feedForward(input)            } -> std::same_as<Eigen::VectorXd>;
  { t.backPropagate(outputGradient) } -> std::same_as<Eigen::RowVectorXd>;

  { std::as_const(t).inputSize()  } -> std::same_as<size_t>;
  { std::as_const(t).outputSize() } -> std::same_as<size_t>;
};
